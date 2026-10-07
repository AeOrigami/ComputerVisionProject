"""Portable review handoffs using SharedStore's schema and trusted master receipts."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import time
import uuid
from contextlib import closing, contextmanager
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from .annotation_database import RUBRIC_FIELDS, SharedStore, identity, timestamp
from .annotation_packages import read_only
from .annotation_schema import Annotation, AnnotationError, validate_annotation

CORE = ("images", "batches", "batch_images", "annotations", "reviews", "reservations", "drafts")
DDL = {
    "review_project": "CREATE TABLE review_project (id INTEGER PRIMARY KEY CHECK(id=1), project_id TEXT NOT NULL)",
    "review_issued": "CREATE TABLE review_issued (snapshot_id TEXT PRIMARY KEY, manifest_sha256 TEXT NOT NULL, created_at TEXT NOT NULL)",
    "review_imports": "CREATE TABLE review_imports (snapshot_id TEXT NOT NULL REFERENCES review_issued, local_review_id INTEGER NOT NULL, master_review_id INTEGER NOT NULL UNIQUE REFERENCES reviews, event_sha256 TEXT NOT NULL, PRIMARY KEY(snapshot_id, local_review_id))",
    "review_snapshot": "CREATE TABLE review_snapshot (id INTEGER PRIMARY KEY CHECK(id=1), manifest TEXT NOT NULL)",
}
MASTER_TABLES = {"review_project", "review_issued", "review_imports"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise AnnotationError(message)


def schema(db):
    return {
        r["name"]: [r["type"], r["tbl_name"], " ".join(r["sql"].split())]
        for r in db.execute("SELECT * FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'")
    }


@lru_cache(maxsize=1)
def expected_core_schema():
    # Derive the format from SharedStore rather than maintain another core schema.
    with tempfile.TemporaryDirectory(prefix="sensifake-schema-") as directory:
        store = SharedStore(Path(directory) / "schema.sqlite3")
        with store.connection() as db:
            return schema(db)


def validate_schema(db, snapshot=False):
    actual = schema(db)
    expected = dict(expected_core_schema())
    allowed = {"review_snapshot"} if snapshot else MASTER_TABLES
    present = allowed & actual.keys()
    require(snapshot or not present or present == MASTER_TABLES, "Incomplete master lineage schema")
    for name in present:
        expected[name] = ["table", name, " ".join(DDL[name].split())]
    require(not snapshot or "review_snapshot" in present, "Not a review snapshot")
    require(actual == expected, "Unexpected database schema, index, view, or trigger")
    require(db.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "SQLite integrity failed")
    require(not db.execute("PRAGMA foreign_key_check").fetchall(), "SQLite foreign keys failed")


def moment(value):
    try:
        result = datetime.fromisoformat(value)
        require(result.tzinfo is not None, "Timestamp must include timezone")
        return result
    except (TypeError, ValueError) as exc:
        raise AnnotationError(f"Invalid timestamp: {value!r}") from exc


def payload(value, content_hash):
    require(isinstance(value, dict), f"Invalid payload: {content_hash}")
    try:
        annotation = Annotation(**value)
        validate_annotation(annotation)
        for key in (
            "public_relevance",
            "harm_urgency",
            "vulnerability",
            "sensitivity_score",
            "annotation_round",
        ):
            require(type(value[key]) is int, f"Invalid integer {key}: {content_hash}")
        require(value["content_hash"] == content_hash, f"Payload hash mismatch: {content_hash}")
    except (TypeError, KeyError, ValueError) as exc:
        raise AnnotationError(f"Invalid annotation payload {content_hash}: {exc}") from exc


def state(db, snapshot=False):
    """Validate and fingerprint durable state; never include draft/lease or image bytes in JSON."""
    validate_schema(db, snapshot)
    result = {"images": []}
    for row in db.execute("SELECT * FROM images ORDER BY content_hash"):
        h = row["content_hash"]
        require(
            hashlib.sha256(row["image_bytes"]).hexdigest() == h, f"Image bytes/hash mismatch: {h}"
        )
        result["images"].append(h)
    for table, order in [
        ("batches", "batch_id"),
        ("batch_images", "batch_id, original_filename"),
        ("annotations", "content_hash"),
        ("reviews", "review_id"),
    ]:
        result[table] = []
        for row in db.execute(f"SELECT * FROM {table} ORDER BY {order}"):
            item = dict(row)
            for key in ("payload", "provenance_json"):
                if key in item:
                    item[key] = json.loads(item[key])
                    require(isinstance(item[key], dict), f"Invalid {key} in {table}")
            result[table].append(item)
    originals = {r["content_hash"]: r for r in result["annotations"]}
    for h, row in originals.items():
        payload(row["payload"], h)
        require(
            identity(row["annotator"]) == (row["annotator_key"], row["annotator"]),
            f"Invalid annotator identity: {h}",
        )
    latest = {h: row["payload"] for h, row in originals.items()}
    dates = {h: moment(row["payload"]["annotated_at"]) for h, row in originals.items()}
    for event in result["reviews"]:
        h, rid = event["content_hash"], event["review_id"]
        label = f"{h}, review {rid}"
        require(type(rid) is int and rid > 0, f"Invalid review ID: {label}")
        require(h in originals, f"Review without original: {label}")
        payload(event["payload"], h)
        require(
            identity(event["reviewer"]) == (event["reviewer_key"], event["reviewer"]),
            f"Invalid reviewer: {label}",
        )
        require(event["reviewer_key"] != originals[h]["annotator_key"], f"Self-review: {label}")
        require(event["action"] in ("confirm", "correct"), f"Invalid action: {label}")
        require(isinstance(event["reason"], str), f"Invalid review reason: {label}")
        require(
            event["action"] != "correct" or event["reason"].strip(),
            f"Missing correction reason: {label}",
        )
        date = moment(event["reviewed_at"])
        require(date >= dates[h], f"Review timestamp runs backward: {label}")
        expected = dict(latest[h])
        if event["action"] == "correct":
            expected.update({key: event["payload"][key] for key in RUBRIC_FIELDS})
        require(event["payload"] == expected, f"Invalid review chain: {label}")
        latest[h], dates[h] = event["payload"], date
    return result


@contextmanager
def writer(path):
    require(Path(path).is_file(), f"Master does not exist: {path}")
    with closing(
        sqlite3.connect(
            Path(path).resolve().as_uri() + "?mode=rw", uri=True, isolation_level=None, timeout=30
        )
    ) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("BEGIN IMMEDIATE")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise


def create_snapshot(master, output_dir):
    """Register an issued base in the master and publish a consistent portable copy."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_id = uuid.uuid4().hex
    destination = output_dir / f"review-{snapshot_id}.sqlite3"
    with tempfile.TemporaryDirectory(prefix=".review-", dir=output_dir) as directory:
        temporary = Path(directory) / "snapshot.sqlite3"
        published = False
        try:
            with writer(master) as db:
                validate_schema(db)
                # No writes before backup: a second read connection sees exactly this locked state.
                with (
                    closing(read_only(Path(master))) as source,
                    closing(sqlite3.connect(temporary)) as copy,
                ):
                    source.backup(copy)
                with closing(read_only(temporary)) as copied:
                    base = state(copied)
                if "review_project" not in schema(db):
                    for name in sorted(MASTER_TABLES):
                        db.execute(DDL[name])
                    db.execute("INSERT INTO review_project VALUES (1,?)", (uuid.uuid4().hex,))
                project_id = db.execute(
                    "SELECT project_id FROM review_project WHERE id=1"
                ).fetchone()[0]
                manifest = {
                    "version": 1,
                    "snapshot_id": snapshot_id,
                    "project_id": project_id,
                    "created_at": timestamp(),
                    "base_sha256": digest(base),
                    "base": base,
                }
                with closing(sqlite3.connect(temporary)) as copy:
                    for name in ("review_imports", "review_issued", "review_project"):
                        copy.execute(f"DROP TABLE IF EXISTS {name}")
                    copy.execute("DELETE FROM reservations")
                    copy.execute("DELETE FROM drafts")
                    copy.execute(DDL["review_snapshot"])
                    copy.execute("INSERT INTO review_snapshot VALUES (1,?)", (canonical(manifest),))
                    copy.commit()
                    copy.execute("PRAGMA journal_mode=DELETE")
                inspect_snapshot(temporary)
                db.execute(
                    "INSERT INTO review_issued VALUES (?,?,?)",
                    (snapshot_id, digest(manifest), manifest["created_at"]),
                )
                with temporary.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.link(temporary, destination)
                published = True
            return destination
        except BaseException:
            if published:
                destination.unlink(missing_ok=True)
            raise


def read_snapshot(path):
    """Read one consistent returned state before touching the master."""
    with closing(read_only(Path(path))) as db:
        db.execute("BEGIN")
        current = state(db, snapshot=True)
        rows = db.execute("SELECT manifest FROM review_snapshot").fetchall()
        require(len(rows) == 1, "Snapshot must have one manifest")
        manifest = json.loads(rows[0][0])
    require(
        set(manifest)
        == {"version", "snapshot_id", "project_id", "created_at", "base_sha256", "base"},
        "Unexpected snapshot metadata",
    )
    require(manifest["version"] == 1, "Unsupported snapshot version")
    for key in ("snapshot_id", "project_id"):
        require(isinstance(manifest[key], str), f"Invalid {key}")
        require(uuid.UUID(manifest[key]).hex == manifest[key], f"Invalid {key}")
    created = moment(manifest["created_at"])
    base = manifest["base"]
    require(digest(base) == manifest["base_sha256"], "Base state digest mismatch")
    for table in ("images", "batches", "batch_images", "annotations"):
        require(base[table] == current[table], f"Snapshot base {table} changed")
    base_events = base["reviews"]
    require(
        current["reviews"][: len(base_events)] == base_events,
        "Snapshot base review history changed",
    )
    new = current["reviews"][len(base_events) :]
    reviewed = {r["content_hash"] for r in base_events}
    seen = set()
    max_id = max((r["review_id"] for r in base_events), default=0)
    for event in new:
        h = event["content_hash"]
        require(
            event["review_id"] > max_id, f"Event ID predates snapshot: {h}, {event['review_id']}"
        )
        require(h not in reviewed and h not in seen, f"Snapshot re-review is not supported: {h}")
        require(moment(event["reviewed_at"]) >= created, f"Review predates snapshot: {h}")
        seen.add(h)
    return manifest, current, new


def summary(manifest, current, new):
    annotated = len(current["annotations"])
    reviewed = len({r["content_hash"] for r in current["reviews"]})
    return {
        "snapshot_id": manifest["snapshot_id"],
        "project_id": manifest["project_id"],
        "created_at": manifest["created_at"],
        "base_sha256": manifest["base_sha256"],
        "images": len(current["images"]),
        "annotated": annotated,
        "unannotated": len(current["images"]) - annotated,
        "reviewed": reviewed,
        "awaiting_review": annotated - reviewed,
        "new_events": len(new),
    }


def plan(db, manifest, new):
    current = state(db)
    require(MASTER_TABLES <= schema(db).keys(), "Master has no issued snapshot registry")
    project = db.execute("SELECT project_id FROM review_project WHERE id=1").fetchone()
    require(project and project[0] == manifest["project_id"], "Wrong master project")
    receipt = db.execute(
        "SELECT manifest_sha256 FROM review_issued WHERE snapshot_id=?", (manifest["snapshot_id"],)
    ).fetchone()
    require(receipt and receipt[0] == digest(manifest), "Unknown snapshot or altered lineage")
    base = manifest["base"]
    require(set(base["images"]) <= set(current["images"]), "Master lost base images")
    for table, keys in [
        ("annotations", ("content_hash",)),
        ("batch_images", ("batch_id", "original_filename")),
        ("batches", ("batch_id",)),
        ("reviews", ("review_id",)),
    ]:
        indexed = {tuple(r[k] for k in keys): dict(r) for r in current[table]}
        for record in base[table]:
            key = tuple(record[k] for k in keys)
            existing = indexed.get(key)
            expected = dict(record)
            if table == "batches" and existing:
                existing.pop("is_open")
                expected.pop("is_open")
            require(existing == expected, f"Master base {table} changed: {key}")
    by_id = {e["review_id"]: e for e in current["reviews"]}
    reviewed = {e["content_hash"] for e in current["reviews"]}
    pending, identical, conflicts = [], 0, []
    for event in new:
        h, rid = event["content_hash"], event["review_id"]
        existing = db.execute(
            "SELECT * FROM review_imports WHERE snapshot_id=? AND local_review_id=?",
            (manifest["snapshot_id"], rid),
        ).fetchone()
        if existing:
            imported = dict(by_id.get(existing["master_review_id"], {}))
            imported["review_id"] = rid
            if existing["event_sha256"] != digest(event) or imported != event:
                conflicts.append(f"event {manifest['snapshot_id']}:{rid} ({h}) changed")
            else:
                identical += 1
        elif h in reviewed:
            conflicts.append(f"{h}: master already has a competing review")
        elif db.execute(
            "SELECT 1 FROM reservations WHERE content_hash=? AND kind='review' AND expires_at>?",
            (h, time.time()),
        ).fetchone():
            conflicts.append(f"{h}: active master review reservation")
        else:
            pending.append(event)
    require(not conflicts, "Merge conflicts: " + "; ".join(conflicts))
    return pending, identical


def inspect_snapshot(path, master=None):
    manifest, current, new = read_snapshot(path)
    result = summary(manifest, current, new)
    result["lineage_verified"] = False
    if master is not None:
        with closing(read_only(Path(master))) as db:
            db.execute("BEGIN")
            pending, identical = plan(db, manifest, new)
        result.update(lineage_verified=True, to_import=len(pending), already_present=identical)
    return result


def merge_snapshot(master, returned):
    manifest, _, new = read_snapshot(returned)
    with writer(master) as db:
        # Lock and validate the current master before the first write; no TOCTOU gap.
        pending, identical = plan(db, manifest, new)
        for event in pending:
            cursor = db.execute(
                "INSERT INTO reviews(content_hash,reviewer_key,reviewer,reviewed_at,action,reason,payload) "
                "VALUES (?,?,?,?,?,?,?)",
                (
                    event["content_hash"],
                    event["reviewer_key"],
                    event["reviewer"],
                    event["reviewed_at"],
                    event["action"],
                    event["reason"],
                    canonical(event["payload"]),
                ),
            )
            db.execute(
                "INSERT INTO review_imports VALUES (?,?,?,?)",
                (manifest["snapshot_id"], event["review_id"], cursor.lastrowid, digest(event)),
            )
        # Validate the entire resulting durable state before committing the transaction.
        state(db)
    return {
        "snapshot_id": manifest["snapshot_id"],
        "new_reviews_imported": len(pending),
        "already_present_identical": identical,
    }
