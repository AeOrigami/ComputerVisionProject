"""Validate ZIP batches and the authoritative legacy snapshot before writing storage."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import stat
import uuid
import warnings
import zipfile
from dataclasses import asdict
from pathlib import Path, PurePosixPath

from PIL import Image, UnidentifiedImageError

from legacy.script_legacy.human_train_assignment import COMPONENT_PATHS

from .annotation_schema import ANNOTATION_FIELDS, AnnotationError, _annotation_from_row

SNAPSHOT = "annotation_data/human-annotation/annotated/total_annotation_401/sensitivity_annotations_401.csv"
SNAPSHOT_SHA256 = "d4cf41d8df87475321f616dfa09b256513c1ee1f380bb2633a33ac693fa2170b"
MAX_IMAGE_BYTES = 40 * 1024 * 1024
MAX_BATCH_BYTES = 512 * 1024 * 1024
MAX_ENTRIES = 10000


def safe_archive_path(name: str) -> str:
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or "\x00" in name
        or ":" in name
        or path.is_absolute()
        or any(part in ("..", ".", "") for part in name.rstrip("/").split("/"))
    ):
        raise AnnotationError(f"Unsafe ZIP path: {name!r}")
    return str(path)


def validate_image(data: bytes, filename: str) -> str:
    if len(data) > MAX_IMAGE_BYTES:
        raise AnnotationError(f"Image exceeds the 40 MiB limit: {filename}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                if getattr(image, "n_frames", 1) != 1:
                    raise AnnotationError(f"Use a single-frame image: {filename}")
                image.load()
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise AnnotationError(f"Unreadable or oversized image: {filename}") from exc
    return hashlib.sha256(data).hexdigest()


def prepare_zip(upload, name: str, dataset_role: str = "custom") -> list[dict]:
    """Optional root metadata.json maps exact archive filenames to provenance objects."""
    if dataset_role not in (
        "custom",
        "rrdataset",
        "gold_development",
        "human_train",
        "model_review",
    ):
        raise AnnotationError("Unknown dataset role.")
    try:
        with zipfile.ZipFile(upload) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_ENTRIES or sum(e.file_size for e in entries) > MAX_BATCH_BYTES:
                raise AnnotationError("ZIP exceeds 10,000 entries or 512 MiB uncompressed.")
            seen = set()
            for entry in entries:
                path = safe_archive_path(entry.filename)
                mode = entry.external_attr >> 16
                if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                    raise AnnotationError(f"ZIP links and special files are not supported: {path}")
                if path in seen:
                    raise AnnotationError(f"Repeated ZIP filename: {path}")
                seen.add(path)
                if entry.flag_bits & 1:
                    raise AnnotationError("Encrypted ZIP entries are not supported.")
            metadata = {}
            if "metadata.json" in seen:
                if archive.getinfo("metadata.json").file_size > 2 * 1024 * 1024:
                    raise AnnotationError("metadata.json exceeds 2 MiB.")
                metadata = json.loads(archive.read("metadata.json"))
                if not isinstance(metadata, dict) or any(
                    not isinstance(v, dict) for v in metadata.values()
                ):
                    raise AnnotationError("metadata.json must map filenames to provenance objects.")
            filenames = {
                e.filename for e in entries if not e.is_dir() and e.filename != "metadata.json"
            }
            if metadata.keys() - filenames:
                raise AnnotationError("Metadata references files absent from the ZIP.")
            model_entries = {
                filename
                for filename, values in metadata.items()
                if values.get("review_type") == "model_review"
            }
            if model_entries:
                if model_entries != filenames:
                    raise AnnotationError("A model-review ZIP needs metadata for every image.")
                dataset_role = "model_review"
            elif dataset_role == "model_review":
                raise AnnotationError("A model-review ZIP needs a model-review metadata.json.")
            images = []
            for entry in entries:
                if entry.is_dir() or entry.filename == "metadata.json":
                    continue
                if entry.file_size > MAX_IMAGE_BYTES:
                    raise AnnotationError(f"Image exceeds 40 MiB: {entry.filename}")
                data = archive.read(entry)
                content_hash = validate_image(data, entry.filename)
                provenance = dict(metadata.get(entry.filename, {}))
                if dataset_role == "model_review":
                    if provenance.get("content_hash", content_hash) != content_hash:
                        raise AnnotationError(f"Model-review hash mismatch: {entry.filename}")
                    if provenance.get("image_id", content_hash) != content_hash:
                        raise AnnotationError(f"Model-review image_id mismatch: {entry.filename}")
                    if provenance.get("image_path", entry.filename) != entry.filename:
                        raise AnnotationError(f"Model-review image_path mismatch: {entry.filename}")
                    if provenance.get("model_prediction") not in ("low", "medium", "high"):
                        raise AnnotationError(f"Missing/invalid model prediction: {entry.filename}")
                    assigned = provenance.get("assigned_annotator")
                    if assigned is not None and (
                        not isinstance(assigned, str) or not assigned.strip() or len(assigned) > 80
                    ):
                        raise AnnotationError(f"Invalid assigned annotator: {entry.filename}")
                    for field in ("confidence", "prob_low", "prob_medium", "prob_high"):
                        value = provenance.get(field, "")
                        if value not in (None, ""):
                            try:
                                number = float(value)
                            except (TypeError, ValueError) as exc:
                                raise AnnotationError(f"Invalid {field}: {entry.filename}") from exc
                            if not math.isfinite(number) or not 0 <= number <= 1:
                                raise AnnotationError(f"Invalid {field}: {entry.filename}")
                    probabilities = [
                        provenance.get(f"prob_{level}", "") for level in ("low", "medium", "high")
                    ]
                    if all(value not in (None, "") for value in probabilities):
                        numbers = [float(value) for value in probabilities]
                        if not math.isclose(sum(numbers), 1, abs_tol=1e-5):
                            raise AnnotationError(
                                f"Model probabilities do not sum to one: {entry.filename}"
                            )
                        predicted = ("low", "medium", "high")[numbers.index(max(numbers))]
                        if predicted != provenance["model_prediction"]:
                            raise AnnotationError(
                                f"Model prediction/probability mismatch: {entry.filename}"
                            )
                        confidence = provenance.get("confidence", "")
                        if confidence not in (None, "") and not math.isclose(
                            float(confidence), max(numbers), abs_tol=1e-5
                        ):
                            raise AnnotationError(f"Model confidence mismatch: {entry.filename}")
                    provenance.setdefault("image_id", content_hash)
                    provenance.setdefault("content_hash", content_hash)
                    provenance.setdefault("image_path", entry.filename)
                    provenance.setdefault("source_dataset", "unknown")
                    label = provenance.get(
                        "normalized_label", provenance.get("source_label", "unknown")
                    )
                    if provenance.get("source_label", label) != label:
                        raise AnnotationError(f"Conflicting source labels: {entry.filename}")
                    if label != "unknown":
                        provenance.setdefault("label_source", "review_manifest_declared")
                else:
                    label = provenance.get("normalized_label", "unknown")
                if label not in ("real", "fake", "unknown"):
                    raise AnnotationError("normalized_label must be real, fake, or unknown.")
                if label != "unknown" and not provenance.get("label_source"):
                    raise AnnotationError(
                        "A real/fake label requires an authoritative label_source."
                    )
                provenance["normalized_label"] = label
                images.append(
                    {
                        "content_hash": content_hash,
                        "image_bytes": data,
                        "original_filename": entry.filename,
                        "provenance": provenance,
                    }
                )
            if not images:
                raise AnnotationError("The ZIP contains no images.")
            return [
                {
                    "batch_id": uuid.uuid4().hex,
                    "name": name,
                    "dataset_role": dataset_role,
                    "images": images,
                }
            ]
    except (zipfile.BadZipFile, json.JSONDecodeError, UnicodeDecodeError, RuntimeError) as exc:
        raise AnnotationError(f"Invalid ZIP or metadata: {exc}") from exc


def prepare_legacy(repository_root: Path) -> list[dict]:
    """Import only the pinned, audited 401 rows; never scan draft or prediction folders."""
    raw = (repository_root / SNAPSHOT).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SNAPSHOT_SHA256:
        raise AnnotationError(
            "The 401-row snapshot differs from the audited SHA-256; import stopped."
        )
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    if len(rows) != 401 or len({r["content_hash"] for r in rows}) != 401:
        raise AnnotationError("Expected exactly 401 unique snapshot images.")
    batches = {
        role: {
            "batch_id": f"legacy-{role}",
            "name": f"Legacy {role}",
            "dataset_role": role,
            "images": [],
        }
        for role in ("gold_development", "human_train")
    }
    for row in rows:
        component_root = (repository_root / COMPONENT_PATHS[row["component"]]).resolve()
        image_path = (component_root / safe_archive_path(row["relative_image_path"])).resolve()
        if not image_path.is_relative_to(repository_root.resolve()):
            raise AnnotationError("Legacy image path escapes its collection directory.")
        data = image_path.read_bytes()
        if validate_image(data, row["relative_image_path"]) != row["content_hash"]:
            raise AnnotationError(f"Legacy image hash mismatch: {row['content_hash']}")
        annotation = asdict(_annotation_from_row({k: row[k] for k in ANNOTATION_FIELDS}, 0))
        provenance = {k: v for k, v in row.items() if k not in ANNOTATION_FIELDS}
        provenance["label_source"] = row["source_dataset"]
        batches[row["dataset_role"]]["images"].append(
            {
                "content_hash": row["content_hash"],
                "image_bytes": data,
                "original_filename": f"{row['component']}/{row['relative_image_path']}",
                "provenance": provenance,
                "annotation": annotation,
                "annotator": row["annotator_id"],
            }
        )
    return list(batches.values())
