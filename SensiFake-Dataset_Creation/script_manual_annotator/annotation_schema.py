"""Validated manifest loading and crash-safe sensitivity annotation storage."""

from __future__ import annotations

import csv
import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ANNOTATION_FIELDS = (
    "content_hash",
    "blind_id",
    "public_relevance",
    "harm_urgency",
    "vulnerability",
    "sensitivity_score",
    "sensitivity_level",
    "sensitivity_rationale",
    "annotation_confidence",
    "needs_review",
    "annotation_round",
    "annotated_at",
)
CONFIDENCE_VALUES = {"high", "medium", "low"}


class ManifestError(RuntimeError):
    """Raised when the authoritative manifest cannot be used safely."""


class AnnotationError(ValueError):
    """Raised when annotation content or storage is invalid."""


class DuplicateAnnotationError(AnnotationError):
    """Raised when a CSV contains a duplicate sample/round key."""


class OverwriteConfirmationRequired(AnnotationError):
    """Raised before replacing an existing non-empty annotation."""


@dataclass(frozen=True)
class BlindedSample:
    """Only the data that annotation mode is permitted to receive."""

    content_hash: str
    image_path: Path
    blind_id: str = ""


@dataclass(frozen=True)
class ReportMetadata:
    """Manifest fields available exclusively to gated presentation mode."""

    authenticity: str
    source_dataset: str


@dataclass(frozen=True)
class ManifestDataset:
    """Separated blinded samples and presentation-only metadata."""

    manifest_path: Path
    samples: tuple[BlindedSample, ...]
    report_metadata: dict[str, ReportMetadata]


@dataclass(frozen=True)
class Annotation:
    content_hash: str
    blind_id: str
    public_relevance: int
    harm_urgency: int
    vulnerability: int
    sensitivity_score: int
    sensitivity_level: str
    sensitivity_rationale: str
    annotation_confidence: str
    needs_review: bool
    annotation_round: int
    annotated_at: str

    @property
    def key(self) -> tuple[str, int]:
        return self.content_hash, self.annotation_round


def calculate_score(public_relevance: int, harm_urgency: int, vulnerability: int) -> int:
    """Validate the three dimensions and return their 0–5 sum."""
    values = (public_relevance, harm_urgency, vulnerability)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise AnnotationError("score dimensions must be integers")
    if public_relevance not in range(3):
        raise AnnotationError("public_relevance must be between 0 and 2")
    if harm_urgency not in range(3):
        raise AnnotationError("harm_urgency must be between 0 and 2")
    if vulnerability not in range(2):
        raise AnnotationError("vulnerability must be between 0 and 1")
    return public_relevance + harm_urgency + vulnerability


def sensitivity_level(score: int) -> str:
    """Map a validated 0–5 score to low, medium, or high."""
    if isinstance(score, bool) or not isinstance(score, int) or score not in range(6):
        raise AnnotationError("sensitivity_score must be between 0 and 5")
    if score <= 1:
        return "low"
    if score <= 3:
        return "medium"
    return "high"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def annotation_from_values(
    *,
    content_hash: str,
    blind_id: str,
    public_relevance: int,
    harm_urgency: int,
    vulnerability: int,
    sensitivity_rationale: str,
    annotation_confidence: str,
    needs_review: bool,
    annotation_round: int = 1,
    annotated_at: str | None = None,
) -> Annotation:
    """Construct and fully validate an annotation from manually entered values."""
    score = calculate_score(public_relevance, harm_urgency, vulnerability)
    annotation = Annotation(
        content_hash=content_hash.strip(),
        blind_id=blind_id.strip(),
        public_relevance=public_relevance,
        harm_urgency=harm_urgency,
        vulnerability=vulnerability,
        sensitivity_score=score,
        sensitivity_level=sensitivity_level(score),
        sensitivity_rationale=sensitivity_rationale.strip(),
        annotation_confidence=annotation_confidence.strip().casefold(),
        needs_review=needs_review,
        annotation_round=annotation_round,
        annotated_at=annotated_at or utc_now(),
    )
    validate_annotation(annotation)
    return annotation


def validate_annotation(annotation: Annotation) -> None:
    if not annotation.content_hash:
        raise AnnotationError("content_hash is required")
    if not annotation.blind_id:
        raise AnnotationError("blind_id is required")
    score = calculate_score(
        annotation.public_relevance,
        annotation.harm_urgency,
        annotation.vulnerability,
    )
    if annotation.sensitivity_score != score:
        raise AnnotationError("sensitivity_score does not match its dimensions")
    if annotation.sensitivity_level != sensitivity_level(score):
        raise AnnotationError("sensitivity_level does not match sensitivity_score")
    if len(annotation.sensitivity_rationale) > 280:
        raise AnnotationError("sensitivity_rationale must contain at most 280 characters")
    if annotation.annotation_confidence not in CONFIDENCE_VALUES:
        raise AnnotationError("annotation_confidence must be high, medium, or low")
    if not isinstance(annotation.needs_review, bool):
        raise AnnotationError("needs_review must be true or false")
    if (
        isinstance(annotation.annotation_round, bool)
        or not isinstance(annotation.annotation_round, int)
        or annotation.annotation_round < 1
    ):
        raise AnnotationError("annotation_round must be a positive integer")
    try:
        parsed_at = datetime.fromisoformat(annotation.annotated_at)
    except ValueError as exc:
        raise AnnotationError("annotated_at must be an ISO-8601 timestamp") from exc
    if parsed_at.tzinfo is None:
        raise AnnotationError("annotated_at must include a timezone")


def load_manifest(path: Path) -> ManifestDataset:
    """Load the authoritative JSONL manifest and resolve every image path."""
    manifest_path = path.resolve()
    if not manifest_path.is_file():
        raise ManifestError(f"authoritative manifest is missing: {manifest_path}")
    dataset_root = manifest_path.parent.resolve()
    samples: list[BlindedSample] = []
    report_metadata: dict[str, ReportMetadata] = {}
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ManifestError(f"manifest row {line_number} is not an object")
                content_hash = row.get("content_hash")
                relative_path = row.get("relative_image_path")
                authenticity = row.get("normalized_label")
                source_dataset = row.get("source_dataset")
                if not all(
                    isinstance(value, str) and value
                    for value in (content_hash, relative_path, authenticity, source_dataset)
                ):
                    raise ManifestError(f"manifest row {line_number} lacks required fields")
                if content_hash in report_metadata:
                    raise ManifestError(f"manifest row {line_number} repeats a content hash")
                relative_image = Path(relative_path)
                if relative_image.is_absolute() or ".." in relative_image.parts:
                    raise ManifestError(f"manifest row {line_number} has an unsafe image path")
                unresolved_image_path = dataset_root / relative_image
                try:
                    image_path = unresolved_image_path.resolve(strict=True)
                except OSError as exc:
                    raise ManifestError(
                        f"manifest image is missing: {unresolved_image_path.absolute()}"
                    ) from exc
                repository_root = Path(__file__).resolve().parents[2]
                allowed = image_path.is_relative_to(dataset_root) or (
                    dataset_root.is_relative_to(repository_root)
                    and image_path.is_relative_to(repository_root)
                )
                if not allowed or not image_path.is_file():
                    raise ManifestError(f"manifest row {line_number} has an unsafe image path")
                samples.append(BlindedSample(content_hash=content_hash, image_path=image_path))
                report_metadata[content_hash] = ReportMetadata(
                    authenticity=authenticity,
                    source_dataset=source_dataset,
                )
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"authoritative manifest could not be loaded: {manifest_path}") from exc
    if not samples:
        raise ManifestError("authoritative manifest contains no samples")
    return ManifestDataset(
        manifest_path=manifest_path,
        samples=tuple(samples),
        report_metadata=report_metadata,
    )


def deterministic_order(
    samples: tuple[BlindedSample, ...] | list[BlindedSample],
    *,
    seed: int = 42,
) -> list[BlindedSample]:
    """Shuffle manifest order reproducibly and assign non-revealing IDs."""
    ordered = list(samples)
    random.Random(seed).shuffle(ordered)
    return [replace(sample, blind_id=f"SF-{index:04d}") for index, sample in enumerate(ordered, 1)]


def _parse_bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise AnnotationError("needs_review must be serialized as true or false")


def _annotation_from_row(row: dict[str, str], row_number: int) -> Annotation:
    try:
        annotation = Annotation(
            content_hash=row["content_hash"],
            blind_id=row["blind_id"],
            public_relevance=int(row["public_relevance"]),
            harm_urgency=int(row["harm_urgency"]),
            vulnerability=int(row["vulnerability"]),
            sensitivity_score=int(row["sensitivity_score"]),
            sensitivity_level=row["sensitivity_level"],
            sensitivity_rationale=row["sensitivity_rationale"],
            annotation_confidence=row["annotation_confidence"],
            needs_review=_parse_bool(row["needs_review"]),
            annotation_round=int(row["annotation_round"]),
            annotated_at=row["annotated_at"],
        )
        validate_annotation(annotation)
    except (KeyError, TypeError, ValueError, AnnotationError) as exc:
        raise AnnotationError(f"invalid annotation CSV row {row_number}: {exc}") from exc
    return annotation


def load_annotations(path: Path) -> list[Annotation]:
    """Load, validate, and reject duplicate persisted annotations."""
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != ANNOTATION_FIELDS:
                raise AnnotationError("annotation CSV header does not match the schema")
            annotations = [
                _annotation_from_row(dict(row), row_number)
                for row_number, row in enumerate(reader, start=2)
            ]
    except OSError as exc:
        raise AnnotationError(f"annotation CSV could not be read: {path}") from exc
    keys = [annotation.key for annotation in annotations]
    if len(keys) != len(set(keys)):
        raise DuplicateAnnotationError("annotation CSV contains duplicate sample/round rows")
    return annotations


def _serialized(annotation: Annotation) -> dict[str, Any]:
    row = asdict(annotation)
    row["needs_review"] = "true" if annotation.needs_review else "false"
    return row


def _same_content(first: Annotation, second: Annotation) -> bool:
    first_values = asdict(first)
    second_values = asdict(second)
    first_values.pop("annotated_at")
    second_values.pop("annotated_at")
    return first_values == second_values


def save_annotation(
    path: Path,
    annotation: Annotation,
    *,
    confirm_overwrite: bool = False,
) -> str:
    """Atomically insert or explicitly replace one annotation and return the action."""
    validate_annotation(annotation)
    annotations = load_annotations(path)
    matching_index = next(
        (index for index, existing in enumerate(annotations) if existing.key == annotation.key),
        None,
    )
    if matching_index is not None:
        existing = annotations[matching_index]
        if _same_content(existing, annotation):
            return "unchanged"
        if not confirm_overwrite:
            raise OverwriteConfirmationRequired(
                "confirm overwrite before replacing this existing annotation"
            )
        annotations[matching_index] = annotation
        action = "updated"
    else:
        annotations.append(annotation)
        action = "created"

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=ANNOTATION_FIELDS)
            writer.writeheader()
            writer.writerows(_serialized(item) for item in annotations)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return action


def resume_index(
    samples: list[BlindedSample],
    annotations: list[Annotation],
    *,
    annotation_round: int = 1,
) -> int:
    """Return the first unannotated position without creating duplicates."""
    completed = {
        annotation.content_hash
        for annotation in annotations
        if annotation.annotation_round == annotation_round
    }
    return next(
        (index for index, sample in enumerate(samples) if sample.content_hash not in completed),
        0,
    )


def prepare_reannotation_sample(
    samples: list[BlindedSample],
    *,
    fraction: float = 0.20,
    seed: int = 42,
) -> list[BlindedSample]:
    """Select a deterministic blind subset for a later annotation round."""
    if not 0 < fraction <= 1:
        raise AnnotationError("re-annotation fraction must be in (0, 1]")
    sample_size = round(len(samples) * fraction)
    selected = random.Random(seed).sample(samples, sample_size)
    return sorted(selected, key=lambda sample: sample.blind_id)
