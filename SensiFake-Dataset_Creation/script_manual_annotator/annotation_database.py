"""Transactional storage for one shared Streamlit instance on a persistent local disk."""

from __future__ import annotations

import csv
import io
import json
import sqlite3
import time
import unicodedata
import uuid
from contextlib import closing, contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .annotation_schema import AnnotationError, annotation_from_values

RUBRIC_FIELDS = (
    "public_relevance",
    "harm_urgency",
    "vulnerability",
    "sensitivity_score",
    "sensitivity_level",
    "sensitivity_rationale",
    "annotation_confidence",
    "needs_review",
)
EXPORT_FIELDS = (
    "batch_id",
    "batch_name",
    "dataset_role",
    "content_hash",
    "blind_id",
    "original_filename",
    "source_dataset",
    "normalized_label",
    "provenance_json",
    "annotator",
    "annotated_at",
    "annotation_round",
    "reviewer",
    "reviewed_at",
    "review_action",
    "review_reason",
    "review_id",
    "status",
    *RUBRIC_FIELDS,
    "original_annotation_json",
)
MODEL_REVIEW_EXPORT_FIELDS = (
    *EXPORT_FIELDS,
    "image_id",
    "image_path",
    "source_label",
    "human_initial_label",
    "human_initial_at",
    "model_prediction",
    "model_confidence",
    "prob_low",
    "prob_medium",
    "prob_high",
    "agreement_initial",
    "human_final_label",
    "agreement_final",
    "model_version",
)
SENSITIVITY_LABELS = ("low", "medium", "high")


def identity(name: str) -> tuple[str, str]:
    display = " ".join(unicodedata.normalize("NFKC", name).split())
    if not display or len(display) > 80:
        raise AnnotationError("Enter a name of 1–80 characters.")
    return display.casefold(), display


def timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def rubric(content_hash: str, values: dict) -> dict:
    """Use the established validator; score and level are always derived."""
    return asdict(
        annotation_from_values(
            content_hash=content_hash,
            blind_id=f"SF-{content_hash[:20]}",
            public_relevance=values["public_relevance"],
            harm_urgency=values["harm_urgency"],
            vulnerability=values["vulnerability"],
            sensitivity_rationale=values.get("sensitivity_rationale", ""),
            annotation_confidence=values["annotation_confidence"],
            needs_review=values.get("needs_review", False),
        )
    )


class SharedStore:
    def __init__(self, path: Path, reservation_seconds: int = 900):
        if reservation_seconds < 1:
            raise ValueError("Reservation timeout must be positive.")
        self.path = Path(path).resolve()
        self.reservation_seconds = reservation_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=30)) as setup:
            setup.execute("PRAGMA journal_mode=WAL")
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS images (
                    content_hash TEXT PRIMARY KEY, image_bytes BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS batches (
                    batch_id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE,
                    dataset_role TEXT NOT NULL, created_at TEXT NOT NULL,
                    is_open INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS batch_images (
                    batch_id TEXT REFERENCES batches, content_hash TEXT REFERENCES images,
                    original_filename TEXT NOT NULL, provenance_json TEXT NOT NULL,
                    PRIMARY KEY(batch_id, original_filename)
                );
                CREATE INDEX IF NOT EXISTS batch_hash ON batch_images(content_hash);
                CREATE TABLE IF NOT EXISTS annotations (
                    content_hash TEXT PRIMARY KEY REFERENCES images,
                    annotator_key TEXT NOT NULL, annotator TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS reviews (
                    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content_hash TEXT NOT NULL REFERENCES annotations,
                    reviewer_key TEXT NOT NULL, reviewer TEXT NOT NULL,
                    reviewed_at TEXT NOT NULL, action TEXT NOT NULL,
                    reason TEXT NOT NULL, payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS review_hash ON reviews(content_hash, review_id);
                CREATE TABLE IF NOT EXISTS reservations (
                    content_hash TEXT NOT NULL REFERENCES images, kind TEXT NOT NULL,
                    person_key TEXT NOT NULL, person TEXT NOT NULL,
                    token TEXT NOT NULL UNIQUE, expires_at REAL NOT NULL,
                    PRIMARY KEY(content_hash, kind), UNIQUE(person_key, kind)
                );
                CREATE TABLE IF NOT EXISTS drafts (
                    content_hash TEXT REFERENCES images, kind TEXT NOT NULL,
                    person_key TEXT NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(content_hash, kind, person_key)
                );
            """)

    @contextmanager
    def connection(self, write: bool = False):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        try:
            db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _snapshot_mode(db) -> bool:
        return (
            db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='review_snapshot'"
            ).fetchone()
            is not None
        )

    @property
    def is_review_snapshot(self) -> bool:
        with self.connection() as db:
            return self._snapshot_mode(db)

    def import_batches(self, batches: list[dict]) -> dict:
        """Commit validated image bytes, memberships and optional legacy decisions together."""
        result = {"new_images": 0, "duplicate_images": 0, "entries": 0, "duplicates": []}
        with self.connection(write=True) as db:
            if self._snapshot_mode(db):
                raise AnnotationError("Review snapshots cannot import images or annotations.")
            for batch in batches:
                name = batch["name"].strip()
                if not name or len(name) > 120:
                    raise AnnotationError("Batch name must contain 1–120 characters.")
                if db.execute(
                    "SELECT 1 FROM batches WHERE name=? OR batch_id=?", (name, batch["batch_id"])
                ).fetchone():
                    raise AnnotationError("This batch already exists; use a different batch name.")
                db.execute(
                    "INSERT INTO batches VALUES (?,?,?,?,1)",
                    (batch["batch_id"], name, batch["dataset_role"], timestamp()),
                )
                for item in batch["images"]:
                    content_hash = item["content_hash"]
                    previous = db.execute(
                        "SELECT bi.batch_id, b.dataset_role FROM batch_images bi "
                        "JOIN batches b USING(batch_id) WHERE bi.content_hash=?",
                        (content_hash,),
                    ).fetchall()
                    if previous and (
                        batch["dataset_role"] == "model_review"
                        or any(row["dataset_role"] == "model_review" for row in previous)
                    ):
                        raise AnnotationError(
                            f"Model-review image already belongs to another batch: {content_hash}"
                        )
                    cursor = db.execute(
                        "INSERT OR IGNORE INTO images VALUES (?,?)",
                        (content_hash, item["image_bytes"]),
                    )
                    result["new_images" if cursor.rowcount else "duplicate_images"] += 1
                    if not cursor.rowcount:
                        result["duplicates"].append(
                            {
                                "filename": item["original_filename"],
                                "content_hash": content_hash,
                                "existing_batches": sorted({r[0] for r in previous}),
                            }
                        )
                    db.execute(
                        "INSERT INTO batch_images VALUES (?,?,?,?)",
                        (
                            batch["batch_id"],
                            content_hash,
                            item["original_filename"],
                            json.dumps(item["provenance"], ensure_ascii=False),
                        ),
                    )
                    result["entries"] += 1
                    if item.get("annotation"):
                        key, person = identity(item["annotator"])
                        if db.execute(
                            "SELECT 1 FROM annotations WHERE content_hash=?", (content_hash,)
                        ).fetchone():
                            raise AnnotationError(
                                "Legacy image already has an annotation; import aborted."
                            )
                        db.execute(
                            "INSERT INTO annotations VALUES (?,?,?,?)",
                            (
                                content_hash,
                                key,
                                person,
                                json.dumps(item["annotation"]),
                            ),
                        )
                        db.execute("DELETE FROM reservations WHERE content_hash=?", (content_hash,))
        return result

    def batches(self) -> list[dict]:
        with self.connection() as db:
            return [
                dict(r) for r in db.execute("SELECT * FROM batches ORDER BY created_at, batch_id")
            ]

    def set_open(self, batch_id: str, is_open: bool) -> None:
        with self.connection(write=True) as db:
            if self._snapshot_mode(db):
                raise AnnotationError("Review snapshot batches cannot be changed.")
            db.execute("UPDATE batches SET is_open=? WHERE batch_id=?", (int(is_open), batch_id))

    def reserve(
        self,
        name: str,
        batch_id: str | None = None,
        *,
        kind: str = "annotation",
        include_reviewed: bool = False,
        now: float | None = None,
    ) -> dict | None:
        if kind not in ("annotation", "review", "model_review"):
            raise AnnotationError("Unknown queue.")
        key, person = identity(name)
        now = time.time() if now is None else now
        with self.connection(write=True) as db:
            if self._snapshot_mode(db) and (kind != "review" or include_reviewed):
                raise AnnotationError("Review snapshots support pending reviews only.")
            db.execute("DELETE FROM reservations WHERE expires_at<=?", (now,))
            existing = db.execute(
                "SELECT * FROM reservations WHERE person_key=? AND kind=?", (key, kind)
            ).fetchone()
            if existing:
                return dict(existing)
            eligibility = (
                "NOT EXISTS (SELECT 1 FROM annotations a WHERE a.content_hash=i.content_hash)"
                if kind in ("annotation", "model_review")
                else "EXISTS (SELECT 1 FROM annotations a WHERE a.content_hash=i.content_hash "
                "AND a.annotator_key<>:person)"
            )
            if kind == "model_review":
                eligibility += (
                    " AND NOT EXISTS (SELECT 1 FROM drafts d WHERE d.content_hash=i.content_hash "
                    "AND d.kind='model_review_completed')"
                    " AND (json_extract(i.provenance_json,'$.assigned_annotator') IS NULL "
                    "OR lower(json_extract(i.provenance_json,'$.assigned_annotator'))=:person)"
                )
            role_filter = (
                "b.dataset_role='model_review'"
                if kind == "model_review"
                else "b.dataset_role<>'model_review'"
            )
            if kind == "review" and not include_reviewed:
                eligibility += (
                    " AND NOT EXISTS (SELECT 1 FROM reviews r WHERE r.content_hash=i.content_hash)"
                )
            row = db.execute(
                f"""
                SELECT DISTINCT i.content_hash FROM batch_images i JOIN batches b USING(batch_id)
                WHERE b.is_open=1 AND (:batch IS NULL OR b.batch_id=:batch)
                AND {role_filter} AND {eligibility}
                AND NOT EXISTS (SELECT 1 FROM reservations r
                    WHERE r.content_hash=i.content_hash AND r.kind=:kind)
                ORDER BY EXISTS (SELECT 1 FROM drafts d WHERE d.content_hash=i.content_hash
                    AND d.person_key=:person AND d.kind=:kind) DESC, i.content_hash LIMIT 1
            """,
                {"batch": batch_id, "person": key, "kind": kind},
            ).fetchone()
            if row is None:
                return None
            lease = {
                "content_hash": row[0],
                "kind": kind,
                "person_key": key,
                "person": person,
                "token": uuid.uuid4().hex,
                "expires_at": now + self.reservation_seconds,
            }
            db.execute(
                "INSERT INTO reservations VALUES (:content_hash,:kind,:person_key,:person,"
                ":token,:expires_at)",
                lease,
            )
            return lease

    def _lease(self, db, token: str, name: str, now: float):
        key, _ = identity(name)
        row = db.execute(
            "SELECT * FROM reservations WHERE token=? AND person_key=? AND expires_at>?",
            (token, key, now),
        ).fetchone()
        if row is None:
            raise AnnotationError("Reservation expired or was released. Reserve an image again.")
        return row

    def release(self, token: str, name: str) -> None:
        key, _ = identity(name)
        with self.connection(write=True) as db:
            db.execute("DELETE FROM reservations WHERE token=? AND person_key=?", (token, key))

    def save_draft(self, token: str, name: str, values: dict, *, now: float | None = None) -> None:
        now = time.time() if now is None else now
        with self.connection(write=True) as db:
            lease = self._lease(db, token, name, now)
            payload = rubric(lease["content_hash"], values)
            payload["review_reason"] = values.get("review_reason", "")
            db.execute(
                "INSERT OR REPLACE INTO drafts VALUES (?,?,?,?)",
                (
                    lease["content_hash"],
                    lease["kind"],
                    lease["person_key"],
                    json.dumps(payload),
                ),
            )
            db.execute(
                "UPDATE reservations SET expires_at=? WHERE token=?",
                (now + self.reservation_seconds, token),
            )

    def image(self, content_hash: str) -> bytes:
        with self.connection() as db:
            return db.execute(
                "SELECT image_bytes FROM images WHERE content_hash=?", (content_hash,)
            ).fetchone()[0]

    def decision(self, content_hash: str) -> dict | None:
        with self.connection() as db:
            original = db.execute(
                "SELECT * FROM annotations WHERE content_hash=?", (content_hash,)
            ).fetchone()
            if original is None:
                return None
            reviews = [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM reviews WHERE content_hash=? ORDER BY review_id", (content_hash,)
                )
            ]
            return {
                "annotator": original["annotator"],
                "original": json.loads(original["payload"]),
                "current": json.loads(reviews[-1]["payload"] if reviews else original["payload"]),
                "reviews": reviews,
            }

    def model_review_case(self, content_hash: str) -> dict:
        with self.connection() as db:
            rows = db.execute(
                "SELECT i.original_filename, i.provenance_json FROM batch_images i "
                "JOIN batches b USING(batch_id) WHERE i.content_hash=? "
                "AND b.dataset_role='model_review' ORDER BY b.created_at, b.batch_id",
                (content_hash,),
            ).fetchall()
            if len(rows) != 1:
                raise AnnotationError(f"Expected one model-review membership for {content_hash}.")
            return {
                "image_path": rows[0]["original_filename"],
                "provenance": json.loads(rows[0]["provenance_json"]),
            }

    def draft(self, lease: dict) -> dict | None:
        with self.connection() as db:
            row = db.execute(
                "SELECT payload FROM drafts WHERE content_hash=? AND kind=? AND person_key=?",
                (lease["content_hash"], lease["kind"], lease["person_key"]),
            ).fetchone()
            return json.loads(row[0]) if row else None

    def save_model_review_draft(
        self,
        token: str,
        name: str,
        *,
        initial: str | None = None,
        final: str | None = None,
        now: float | None = None,
    ) -> dict:
        now = time.time() if now is None else now
        with self.connection(write=True) as db:
            lease = self._lease(db, token, name, now)
            if lease["kind"] != "model_review":
                raise AnnotationError("Not a model-review reservation.")
            row = db.execute(
                "SELECT payload FROM drafts WHERE content_hash=? AND kind='model_review' "
                "AND person_key=?",
                (lease["content_hash"], lease["person_key"]),
            ).fetchone()
            payload = json.loads(row[0]) if row else {}
            if initial is not None:
                if initial not in SENSITIVITY_LABELS:
                    raise AnnotationError("Choose low, medium, or high.")
                if payload.get("human_initial_label") not in (None, initial):
                    raise AnnotationError(
                        "The initial human label is already saved and cannot change."
                    )
                if not payload:
                    payload = {"human_initial_label": initial, "initial_at": timestamp()}
            if not payload.get("human_initial_label"):
                raise AnnotationError("Save the initial human label before revealing the model.")
            if final is not None:
                if final not in SENSITIVITY_LABELS:
                    raise AnnotationError("Choose low, medium, or high.")
                payload["human_final_label"] = final
            db.execute(
                "INSERT OR REPLACE INTO drafts VALUES (?,?,?,?)",
                (lease["content_hash"], "model_review", lease["person_key"], json.dumps(payload)),
            )
            db.execute(
                "UPDATE reservations SET expires_at=? WHERE token=?",
                (now + self.reservation_seconds, token),
            )
            return payload

    def complete_model_review(
        self, token: str, name: str, final: str, *, now: float | None = None
    ) -> None:
        if final not in SENSITIVITY_LABELS:
            raise AnnotationError("Choose low, medium, or high.")
        now = time.time() if now is None else now
        with self.connection(write=True) as db:
            lease = self._lease(db, token, name, now)
            if lease["kind"] != "model_review":
                raise AnnotationError("Not a model-review reservation.")
            draft = db.execute(
                "SELECT payload FROM drafts WHERE content_hash=? AND kind='model_review' "
                "AND person_key=?",
                (lease["content_hash"], lease["person_key"]),
            ).fetchone()
            if draft is None:
                raise AnnotationError("The initial human label has not been saved.")
            values = json.loads(draft[0])
            initial = values.get("human_initial_label")
            if initial not in SENSITIVITY_LABELS:
                raise AnnotationError("Invalid saved initial human label.")
            memberships = db.execute(
                "SELECT i.provenance_json FROM batch_images i JOIN batches b USING(batch_id) "
                "WHERE i.content_hash=? AND b.dataset_role='model_review'",
                (lease["content_hash"],),
            ).fetchall()
            if len(memberships) != 1:
                raise AnnotationError("Expected exactly one model-review batch membership.")
            model_prediction = json.loads(memberships[0][0]).get("model_prediction")
            if model_prediction not in SENSITIVITY_LABELS:
                raise AnnotationError("Missing model prediction in batch metadata.")
            if db.execute(
                "SELECT 1 FROM drafts WHERE content_hash=? AND kind='model_review_completed'",
                (lease["content_hash"],),
            ).fetchone():
                raise AnnotationError("This image already has a completed model review.")
            payload = {
                "record_type": "model_review",
                "content_hash": lease["content_hash"],
                "annotator": lease["person"],
                "human_initial_label": initial,
                "initial_at": values["initial_at"],
                "human_final_label": final,
                "agreement_initial": initial == model_prediction,
                "agreement_final": final == model_prediction,
                "annotated_at": timestamp(),
            }
            # Keep completed model reviews outside the rubric-only annotations table.
            # A distinct kind in the existing durable drafts table needs no schema change.
            db.execute(
                "INSERT INTO drafts VALUES (?,?,?,?)",
                (
                    lease["content_hash"],
                    "model_review_completed",
                    lease["person_key"],
                    json.dumps(payload),
                ),
            )
            db.execute("DELETE FROM reservations WHERE token=?", (token,))
            db.execute(
                "DELETE FROM drafts WHERE content_hash=? AND kind='model_review' AND person_key=?",
                (lease["content_hash"], lease["person_key"]),
            )

    def complete(
        self,
        token: str,
        name: str,
        values: dict | None = None,
        *,
        action: str = "confirm",
        reason: str = "",
        now: float | None = None,
    ) -> None:
        now = time.time() if now is None else now
        with self.connection(write=True) as db:
            lease = self._lease(db, token, name, now)
            if lease["kind"] not in ("annotation", "review"):
                raise AnnotationError("Use the model-review completion action for this image.")
            content_hash = lease["content_hash"]
            if self._snapshot_mode(db) and (
                lease["kind"] != "review"
                or db.execute(
                    "SELECT 1 FROM reviews WHERE content_hash=?", (content_hash,)
                ).fetchone()
            ):
                raise AnnotationError("Review snapshots support pending reviews only.")
            if lease["kind"] == "annotation":
                payload = rubric(content_hash, values or {})
                db.execute(
                    "INSERT INTO annotations VALUES (?,?,?,?)",
                    (
                        content_hash,
                        lease["person_key"],
                        lease["person"],
                        json.dumps(payload),
                    ),
                )
            else:
                original = db.execute(
                    "SELECT * FROM annotations WHERE content_hash=?", (content_hash,)
                ).fetchone()
                if original["annotator_key"] == lease["person_key"]:
                    raise AnnotationError("You cannot review your own annotation.")
                if action not in ("confirm", "correct"):
                    raise AnnotationError("Unknown review action.")
                if action == "correct" and not reason.strip():
                    raise AnnotationError("A correction requires a reason.")
                latest = db.execute(
                    "SELECT payload FROM reviews WHERE content_hash=? "
                    "ORDER BY review_id DESC LIMIT 1",
                    (content_hash,),
                ).fetchone()
                payload = json.loads(latest[0] if latest else original["payload"])
                if action == "correct":
                    correction = rubric(content_hash, values or {})
                    payload.update({field: correction[field] for field in RUBRIC_FIELDS})
                db.execute(
                    "INSERT INTO reviews(content_hash,reviewer_key,reviewer,reviewed_at,"
                    "action,reason,payload) VALUES (?,?,?,?,?,?,?)",
                    (
                        content_hash,
                        lease["person_key"],
                        lease["person"],
                        timestamp(),
                        action,
                        reason.strip(),
                        json.dumps(payload),
                    ),
                )
            db.execute("DELETE FROM reservations WHERE token=?", (token,))
            db.execute(
                "DELETE FROM drafts WHERE content_hash=? AND kind=? AND person_key=?",
                (content_hash, lease["kind"], lease["person_key"]),
            )

    def progress(self) -> list[dict]:
        with self.connection() as db:
            rows = []
            for batch in [None, *db.execute("SELECT * FROM batches ORDER BY name").fetchall()]:
                scope = "SELECT DISTINCT content_hash FROM batch_images"
                args = []
                if batch:
                    scope += " WHERE batch_id=?"
                    args.append(batch["batch_id"])
                counts = db.execute(
                    f"""
                    SELECT COUNT(*) total,
                    SUM(EXISTS(SELECT 1 FROM annotations a WHERE a.content_hash=s.content_hash)) annotated,
                    SUM(EXISTS(SELECT 1 FROM reviews r WHERE r.content_hash=s.content_hash)) confirmed,
                    SUM(EXISTS(SELECT 1 FROM reservations r WHERE r.content_hash=s.content_hash
                        AND r.kind IN ('annotation','model_review') AND r.expires_at>?)) reserved
                    FROM ({scope}) s
                """,
                    [time.time(), *args],
                ).fetchone()
                total, annotated, confirmed, reserved = (v or 0 for v in counts)
                if batch is None or batch["dataset_role"] == "model_review":
                    completed = db.execute(
                        f"SELECT COUNT(*) FROM ({scope}) s WHERE EXISTS "
                        "(SELECT 1 FROM drafts d WHERE d.content_hash=s.content_hash "
                        "AND d.kind='model_review_completed')",
                        args,
                    ).fetchone()[0]
                    annotated += completed
                    confirmed += completed
                rows.append(
                    {
                        "batch": batch["name"] if batch else "All batches (unique images)",
                        "batch_id": batch["batch_id"] if batch else "",
                        "total": total,
                        "annotated": annotated,
                        "confirmed": confirmed,
                        "awaiting_review": annotated - confirmed,
                        "unannotated": total - annotated,
                        "reserved": reserved,
                    }
                )
            return rows

    def export_csv(self, kind: str = "current", batch_id: str | None = None) -> bytes:
        if kind == "model_review":
            return self.export_model_review_csv(batch_id)
        if kind not in ("current", "confirmed", "history"):
            raise AnnotationError("Unknown export.")
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=EXPORT_FIELDS)
        writer.writeheader()
        with self.connection() as db:
            for row in db.execute(
                """
                SELECT b.*, i.*, a.annotator, a.payload FROM batch_images i
                JOIN batches b USING(batch_id) JOIN annotations a USING(content_hash)
                WHERE b.dataset_role<>'model_review' AND (? IS NULL OR b.batch_id=?)
                ORDER BY b.batch_id, i.original_filename
            """,
                (batch_id, batch_id),
            ):
                original = json.loads(row["payload"])
                reviews = db.execute(
                    "SELECT * FROM reviews WHERE content_hash=? ORDER BY review_id",
                    (row["content_hash"],),
                ).fetchall()
                selected = reviews if kind == "history" else [reviews[-1] if reviews else None]
                if kind == "confirmed" and not reviews:
                    continue
                provenance = json.loads(row["provenance_json"])
                for review in selected:
                    payload = json.loads(review["payload"]) if review else original
                    record = {
                        "batch_id": row["batch_id"],
                        "batch_name": row["name"],
                        "dataset_role": row["dataset_role"],
                        "original_filename": row["original_filename"],
                        "provenance_json": row["provenance_json"],
                        "source_dataset": provenance.get("source_dataset", "unknown"),
                        "normalized_label": provenance.get("normalized_label", "unknown"),
                        "annotator": row["annotator"],
                        "reviewer": review["reviewer"] if review else "",
                        "reviewed_at": review["reviewed_at"] if review else "",
                        "review_action": review["action"] if review else "",
                        "review_reason": review["reason"] if review else "",
                        "review_id": review["review_id"] if review else "",
                        "status": "confirmed" if review else "awaiting_review",
                        "original_annotation_json": row["payload"],
                    }
                    record.update(
                        {
                            k: payload[k]
                            for k in (
                                *RUBRIC_FIELDS,
                                "content_hash",
                                "blind_id",
                                "annotation_round",
                                "annotated_at",
                            )
                        }
                    )
                    record["needs_review"] = str(record["needs_review"]).lower()
                    writer.writerow(record)
        return output.getvalue().encode("utf-8")

    def export_model_review_csv(self, batch_id: str | None = None) -> bytes:
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=MODEL_REVIEW_EXPORT_FIELDS)
        writer.writeheader()
        with self.connection() as db:
            for row in db.execute(
                "SELECT b.batch_id, b.name, b.dataset_role, i.content_hash, "
                "i.original_filename, i.provenance_json, d.payload "
                "FROM batch_images i JOIN batches b USING(batch_id) "
                "JOIN drafts d USING(content_hash) "
                "WHERE b.dataset_role='model_review' AND (? IS NULL OR b.batch_id=?) "
                "AND d.kind='model_review_completed' "
                "ORDER BY b.batch_id, i.original_filename",
                (batch_id, batch_id),
            ):
                payload = json.loads(row["payload"])
                if payload.get("record_type") != "model_review":
                    raise AnnotationError(f"Invalid model-review decision: {row['content_hash']}")
                provenance = json.loads(row["provenance_json"])
                record = {
                    "batch_id": row["batch_id"],
                    "batch_name": row["name"],
                    "dataset_role": row["dataset_role"],
                    "content_hash": row["content_hash"],
                    "blind_id": f"SF-{row['content_hash'][:20]}",
                    "original_filename": row["original_filename"],
                    "source_dataset": provenance.get("source_dataset", "unknown"),
                    "normalized_label": provenance.get("normalized_label", "unknown"),
                    "provenance_json": row["provenance_json"],
                    "annotator": payload["annotator"],
                    "annotated_at": payload["annotated_at"],
                    "status": "model_review_completed",
                    "image_id": provenance.get("image_id", row["content_hash"]),
                    "image_path": row["original_filename"],
                    "source_label": provenance.get(
                        "source_label", provenance.get("normalized_label", "unknown")
                    ),
                    "human_initial_label": payload["human_initial_label"],
                    "human_initial_at": payload["initial_at"],
                    "model_prediction": provenance["model_prediction"],
                    "model_confidence": provenance.get("confidence", ""),
                    "prob_low": provenance.get("prob_low", ""),
                    "prob_medium": provenance.get("prob_medium", ""),
                    "prob_high": provenance.get("prob_high", ""),
                    "agreement_initial": str(payload["agreement_initial"]).lower(),
                    "human_final_label": payload["human_final_label"],
                    "agreement_final": str(payload["agreement_final"]).lower(),
                    "model_version": provenance.get("model_version", ""),
                    "sensitivity_level": payload["human_final_label"],
                }
                writer.writerow(record)
        return output.getvalue().encode("utf-8")
