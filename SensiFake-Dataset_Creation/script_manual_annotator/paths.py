"""Repository path resolution shared by the annotation app and tests."""

from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def canonical_openfake_pilot_manifest(repository_root: Path = REPOSITORY_ROOT) -> Path:
    """Return the canonical OpenFake pilot manifest path."""
    return repository_root / "annotation_data/source_datasets/openfake/pilot-600/manifest.jsonl"


canonical_openfake_manifest = canonical_openfake_pilot_manifest


def canonical_openfake_additional_manifest(repository_root: Path = REPOSITORY_ROOT) -> Path:
    """Return the canonical OpenFake additional-900 manifest path."""
    return repository_root / "annotation_data/source_datasets/openfake/additional-900/manifest.jsonl"


def canonical_sid_set_manifest(repository_root: Path = REPOSITORY_ROOT) -> Path:
    """Return the canonical SID-Set candidate-1500 manifest path."""
    return repository_root / "annotation_data/source_datasets/sid-set/candidate-1500/manifest.jsonl"


def openfake_development_annotations(repository_root: Path = REPOSITORY_ROOT) -> Path:
    """Return the writable OpenFake development-v0 annotation CSV path."""
    return repository_root / "annotation_data/openfake/development-v0/sensitivity_annotations.csv"


def gold_silver_assignment_path(repository_root: Path = REPOSITORY_ROOT) -> Path:
    """Return the canonical Gold/Silver split assignment CSV path."""
    return repository_root / "annotation_data/splits/gold_silver_assignment.csv"
