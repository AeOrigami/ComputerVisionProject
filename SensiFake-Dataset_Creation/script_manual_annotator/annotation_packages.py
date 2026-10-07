"""Prepare disjoint local annotation databases and reconcile their completed work."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections import defaultdict
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

from .annotation_database import SharedStore, identity
from .annotation_schema import AnnotationError, annotation_from_values


def read_only(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise AnnotationError(f"Database does not exist: {path}")
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def prepare_offline(master_path: Path, batch_name: str, parts: int, output_dir: Path) -> dict:
    """Split unannotated hashes evenly; package only blind filenames and image bytes."""
    if parts < 2:
        raise AnnotationError("Split the batch into at least two parts.")
    with closing(read_only(master_path)) as db:
        batch = db.execute("SELECT batch_id FROM batches WHERE name=?", (batch_name,)).fetchone()
        if batch is None:
            raise AnnotationError(f"Unknown batch: {batch_name}")
        rows = db.execute(
            """
            SELECT bi.content_hash, bi.provenance_json, i.image_bytes
            FROM batch_images bi JOIN images i USING(content_hash)
            WHERE bi.batch_id=? AND NOT EXISTS (
                SELECT 1 FROM annotations a WHERE a.content_hash=bi.content_hash
            ) ORDER BY bi.content_hash
            """,
            (batch["batch_id"],),
        ).fetchall()
        if not rows or len({row["content_hash"] for row in rows}) != len(rows):
            raise AnnotationError("Batch has no unique unannotated images to split.")
        if len(rows) % parts:
            raise AnnotationError("Image count must divide evenly among parts.")
        groups = defaultdict(list)
        for row in rows:
            group = json.loads(row["provenance_json"]).get("selection_group", "")
            groups[group].append(row)
        assignments = [[] for _ in range(parts)]
        index = 0
        for group in sorted(groups):
            for row in groups[group]:
                assignments[index % parts].append(row)
                index += 1
        if output_dir.exists():
            raise AnnotationError(f"Output directory already exists: {output_dir}")
        output_dir.mkdir(parents=True)
        manifest = {"version": 1, "source_batch_id": batch["batch_id"], "packages": []}
        for number, assigned in enumerate(assignments, 1):
            filename = f"annotator-{number:02d}.sqlite3"
            package_id = uuid.uuid4().hex
            images = []
            for row in assigned:
                data = row["image_bytes"]
                content_hash = row["content_hash"]
                if hashlib.sha256(data).hexdigest() != content_hash:
                    raise AnnotationError(f"Stored image hash mismatch: {content_hash}")
                images.append(
                    {
                        "content_hash": content_hash,
                        "image_bytes": data,
                        "original_filename": f"images/{content_hash}",
                        "provenance": {"normalized_label": "unknown"},
                    }
                )
            package = SharedStore(output_dir / filename)
            package.import_batches(
                [{"batch_id": package_id, "name": "My offline images", "dataset_role": "custom", "images": images}]
            )
            manifest["packages"].append(
                {
                    "package": filename,
                    "package_batch_id": package_id,
                    "hashes": sorted(row["content_hash"] for row in assigned),
                }
            )
        (output_dir / "assignment.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return manifest


def snapshot_offline(package_path: Path, output_path: Path) -> None:
    """Create a consistent single-file copy, even while the app uses SQLite WAL."""
    if output_path.exists():
        raise AnnotationError(f"Snapshot already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with closing(read_only(package_path)) as source, closing(
            sqlite3.connect(output_path)
        ) as destination:
            source.backup(destination)
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise


def merge_offline(master_path: Path, manifest_path: Path, package_paths: list[Path]) -> int:
    """Import valid completed annotations atomically; identical retries are harmless."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != 1:
        raise AnnotationError("Unsupported offline assignment manifest.")
    packages = {entry["package_batch_id"]: entry for entry in manifest["packages"]}
    if len(packages) != len(manifest["packages"]):
        raise AnnotationError("Duplicate package ID in assignment manifest.")
    if not package_paths:
        raise AnnotationError("Provide at least one returned package.")
    incoming = {}
    seen_packages = set()
    for path in package_paths:
        with closing(read_only(path)) as db:
            batch_ids = [row[0] for row in db.execute("SELECT batch_id FROM batches")]
            if len(batch_ids) != 1 or batch_ids[0] not in packages:
                raise AnnotationError(f"Unexpected offline package: {path}")
            package_id = batch_ids[0]
            if package_id in seen_packages:
                raise AnnotationError("The same offline assignment was supplied twice.")
            seen_packages.add(package_id)
            entry = packages[package_id]
            expected_hashes = set(entry["hashes"])
            actual_hashes = {
                row[0] for row in db.execute("SELECT content_hash FROM batch_images")
            }
            if actual_hashes != expected_hashes:
                raise AnnotationError(f"Offline image assignment changed: {path}")
            for row in db.execute("SELECT * FROM annotations"):
                content_hash = row["content_hash"]
                if content_hash not in expected_hashes:
                    raise AnnotationError(f"Unexpected image in {path}")
                if row["annotator_key"] != identity(row["annotator"])[0]:
                    raise AnnotationError(f"Invalid annotator identity in {path}")
                payload = json.loads(row["payload"])
                validated = asdict(
                    annotation_from_values(
                        content_hash=content_hash,
                        blind_id=f"SF-{content_hash[:20]}",
                        public_relevance=payload["public_relevance"],
                        harm_urgency=payload["harm_urgency"],
                        vulnerability=payload["vulnerability"],
                        sensitivity_rationale=payload["sensitivity_rationale"],
                        annotation_confidence=payload["annotation_confidence"],
                        needs_review=payload["needs_review"],
                        annotation_round=payload["annotation_round"],
                        annotated_at=payload["annotated_at"],
                    )
                )
                if payload != validated or payload["annotation_round"] != 1:
                    raise AnnotationError(f"Invalid offline annotation: {content_hash}")
                incoming[content_hash] = (row["annotator_key"], row["annotator"], row["payload"])
    store = SharedStore(master_path)
    inserted = 0
    with store.connection(write=True) as db:
        source_hashes = {
            row[0]
            for row in db.execute(
                "SELECT content_hash FROM batch_images WHERE batch_id=?",
                (manifest["source_batch_id"],),
            )
        }
        if not source_hashes or any(hash_value not in source_hashes for hash_value in incoming):
            raise AnnotationError("Master batch differs from offline assignment.")
        for content_hash, values in incoming.items():
            existing = db.execute(
                "SELECT annotator_key,annotator,payload FROM annotations WHERE content_hash=?",
                (content_hash,),
            ).fetchone()
            if existing:
                if tuple(existing) != values:
                    raise AnnotationError(f"Conflicting annotation: {content_hash}")
                continue
            db.execute(
                "INSERT INTO annotations VALUES (?,?,?,?)",
                (content_hash, *values),
            )
            inserted += 1
    return inserted
