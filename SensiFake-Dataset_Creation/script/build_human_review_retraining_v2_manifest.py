"""Build or validate the V2 human-review manifest without replacing existing artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path("model_autoannotator/inputs/sensitivity_resnet50_final")
MANIFEST = Path("model_autoannotator/inputs/v2/human_review_retraining_v2_manifest.csv")
AUDIT = MANIFEST.with_suffix(".audit.json")
REVIEW = Path("annotation_data/model-autoannotation/human_revision/review_sara.csv")
MASTER = Path("annotation_data/human-annotation/master/reviewed/human_annotations_master_final.csv")
HISTORICAL = Path("annotation_data/model-autoannotation/human_revision/sensifake_model_review.csv")
REFERENCE = Path("annotation_data/model-autoannotation/human_revision/model_review_giovanni.csv")
SUPERSEDED = Path("model_autoannotator/inputs/v2/human_review_retraining_v2_superseded_labels.csv")
NOTEBOOK = Path("model_autoannotator/notebooks/experiments/sensitivity_resnet50_final_kaggle.ipynb")
CHECKPOINT = Path("model_autoannotator/notebooks/experiments/final output/best_model.pt")
PREVIOUS_CONFIG = Path("model_autoannotator/notebooks/experiments/final output/config.json")
CLASSES = ("low", "medium", "high")
FIELDS = (
    "content_hash",
    "image_filename",
    "final_sensitivity_level",
    "source_dataset",
    "source_label",
    "source_label_provenance",
    "split",
    "internal_split",
    "image_id",
    "image_path",
    "sensitivity_level",
    "annotation_source",
    "annotation_file",
    "annotation_row",
    "annotator",
    "annotated_at",
    "review_status",
    "human_final_label",
    "human_initial_label",
    "model_prediction",
    "model_version",
    "provenance_json",
    "source_paths_json",
    "split_seed",
    "split_strategy",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        require(bool(fields) and len(fields) == len(set(fields)), f"Invalid CSV header: {path}")
        rows = list(reader)
    require(
        all(None not in row and None not in row.values() for row in rows), f"Malformed CSV: {path}"
    )
    return rows


def unique_hashes(rows: list[dict], name: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        digest = row.get("content_hash", "")
        require(bool(re.fullmatch(r"[0-9a-f]{64}", digest)), f"Invalid hash in {name}: {digest}")
        require(digest not in result, f"Duplicate content_hash in {name}: {digest}")
        result[digest] = row
    return result


def csv_bytes(rows: list[dict], fields: tuple | list = FIELDS) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def assign_new_splits(rows: list[dict], seed: int = 42) -> dict[str, str]:
    """Allocate exactly 20% validation across dataset × final-label strata."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["source_dataset"], row["human_final_label"])].append(row["content_hash"])
    target = round(len(rows) * 0.20)
    ideals = {key: len(group) * 0.20 for key, group in groups.items()}
    quotas = {key: math.floor(value) for key, value in ideals.items()}
    for key in sorted(groups, key=lambda key: (-(ideals[key] - quotas[key]), key)):
        if sum(quotas.values()) == target:
            break
        quotas[key] += 1
    require(sum(quotas.values()) == target, "Could not allocate new validation quota")
    rng = random.Random(seed)
    assignments = {}
    for key in sorted(groups):
        hashes = sorted(groups[key])
        rng.shuffle(hashes)
        validation = set(hashes[: quotas[key]])
        for digest in hashes:
            assignments[digest] = "validation" if digest in validation else "effective_train"
    return assignments


def distributions(rows: list[dict]) -> dict:
    def counts(group):
        counter = Counter(row["final_sensitivity_level"] for row in group)
        return {label: counter[label] for label in CLASSES}

    sources = sorted({row["source_dataset"] for row in rows})
    return {
        "rows": len(rows),
        "classes": counts(rows),
        "source_dataset": dict(sorted(Counter(row["source_dataset"] for row in rows).items())),
        "classes_by_source_dataset": {
            source: counts([row for row in rows if row["source_dataset"] == source])
            for source in sources
        },
    }


def identity_checks(rows: list[dict], name: str) -> None:
    """Compare IDs, then hashes, then canonical paths; reject inconsistent aliases."""
    for field in ("image_id", "content_hash", "image_path"):
        seen = {}
        for row in rows:
            value = row[field]
            require(bool(value), f"Missing {field} in {name}")
            if field == "image_path":
                value = Path(value).as_posix()
            if value in seen:
                previous = seen[value]
                require(
                    previous["final_sensitivity_level"] == row["final_sensitivity_level"],
                    f"Conflicting labels for {field}={value} in {name}",
                )
                raise ValueError(f"Duplicate {field}={value} in {name}")
            seen[value] = row


def build_rows(root: Path = ROOT) -> tuple[list[dict], dict, list[dict]]:
    package = root / PACKAGE
    package_audit = json.loads((package / "package_audit.json").read_text())
    old_config = json.loads((root / PREVIOUS_CONFIG).read_text())
    require(old_config["seed"] == package_audit["seed"] == 42, "Previous seed changed")
    require(tuple(old_config["class_order"]) == CLASSES, "Previous class order changed")
    for filename, expected in package_audit["csv_sha256"].items():
        require(sha256(package / filename) == expected, f"Previous package changed: {filename}")
    for filename, expected in old_config["package_csv_sha256"].items():
        require(sha256(package / filename) == expected, f"V1 config/package mismatch: {filename}")
    require(
        sha256(root / MASTER) == package_audit["source_sha256"][MASTER.name],
        "Previous human master changed",
    )
    for filename in ("automatic_annotator_train.csv", "automatic_annotator_test.csv"):
        require(
            sha256(root / "annotation_data/splits" / filename)
            == package_audit["source_sha256"][filename],
            f"Frozen split changed: {filename}",
        )

    old_rows = read_csv(package / "train.csv")
    test_rows = read_csv(package / "test.csv")
    old = unique_hashes(old_rows, "previous train/validation")
    test = unique_hashes(test_rows, "fixed test")
    corpus = unique_hashes(read_csv(package / "corpus_manifest.csv"), "corpus")
    master_rows = read_csv(root / MASTER)
    master = unique_hashes(master_rows, "master")
    master_row_numbers = {
        row["content_hash"]: str(number) for number, row in enumerate(master_rows, 2)
    }
    require(len(old) == 561 and len(test) == 140, "Previous split counts changed")
    require(not old.keys() & test.keys(), "Previous train/test overlap")
    require(
        {h for h, row in old.items() if row["internal_split"] == "validation"}
        == set(package_audit["validation_content_hashes"]),
        "Frozen validation changed",
    )

    reviews = read_csv(root / REVIEW)
    require(len(reviews) == 400, f"Expected 400 canonical review records, got {len(reviews)}")
    reviewed = {}
    reviewed_ids = {}
    identical_review_duplicates = []
    for number, row in enumerate(reviews, 2):
        h = row["content_hash"]
        require(bool(re.fullmatch(r"[0-9a-f]{64}", h)), f"Invalid reviewed hash: {h}")
        require(row["human_final_label"] in CLASSES, f"Missing/invalid human final label: {h}")
        require(row["status"] == "model_review_completed", f"Incomplete human review: {h}")
        require(bool(row["image_id"]), f"Missing reviewed image_id: {h}")
        require(
            row["image_id"] not in reviewed_ids or reviewed_ids[row["image_id"]] == h,
            f"Reviewed image_id refers to different hashes: {row['image_id']}",
        )
        reviewed_ids[row["image_id"]] = h
        if h in reviewed:
            require(
                reviewed[h]["human_final_label"] == row["human_final_label"],
                f"Conflicting canonical human_final_label for {h}",
            )
            require(
                all(
                    reviewed[h][key] == row[key]
                    for key in ("image_id", "image_path", "source_dataset", "source_label")
                ),
                f"Ambiguous reviewed identity/source metadata for {h}",
            )
            identical_review_duplicates.append(h)
            continue
        reviewed[h] = dict(row, annotation_row=str(number))

    test_ids = {master[h]["image_id"] for h in test}
    require(not reviewed.keys() & test.keys(), "Canonical reviews overlap the fixed test")
    require(not reviewed_ids.keys() & test_ids, "Canonical reviewed IDs overlap fixed test")
    assignments = assign_new_splits(list(reviewed.values()), seed=old_config["seed"])
    result = {}
    for h, old_row in old.items():
        human = master[h]
        require(
            old_row["final_sensitivity_level"] == human["final_sensitivity_level"],
            f"V1 target/master mismatch: {h}",
        )
        row = dict.fromkeys(FIELDS, "")
        row.update(old_row)
        row.update(
            {
                "image_id": human["image_id"],
                "image_path": (PACKAGE / old_row["image_filename"]).as_posix(),
                "sensitivity_level": old_row["final_sensitivity_level"],
                "annotation_source": human["annotation_source"] or "human",
                "annotation_file": MASTER.as_posix(),
                "annotation_row": master_row_numbers[h],
                "annotator": human["annotator"],
                "annotated_at": human["annotated_at"],
                "review_status": human["review_status"],
                "provenance_json": human["provenance_json"],
                "source_paths_json": corpus[h]["source_paths_json"],
                "split_seed": "42",
                "split_strategy": "preserved_v1_internal_split",
            }
        )
        result[h] = row

    old_ids = {row["image_id"]: h for h, row in result.items()}
    replacements = []
    for h, review in reviewed.items():
        require(h in corpus, f"Reviewed image absent from canonical corpus: {h}")
        source = corpus[h]
        for key in ("source_dataset", "source_label"):
            require(review[key] == source[key], f"Review/corpus {key} conflict for {h}")
        require(
            review["image_path"] == source["image_filename"], f"Review/corpus path conflict: {h}"
        )
        if review["image_id"] in old_ids:
            require(
                old_ids[review["image_id"]] == h, f"Image ID/hash conflict: {review['image_id']}"
            )
        previous = result.get(h)
        if previous:
            replacements.append(
                {
                    "content_hash": h,
                    "previous_label": previous["final_sensitivity_level"],
                    "human_final_label": review["human_final_label"],
                }
            )
        row = dict.fromkeys(FIELDS, "")
        row.update(
            {
                "content_hash": h,
                "image_id": review["image_id"],
                "image_filename": source["image_filename"],
                "image_path": (PACKAGE / source["image_filename"]).as_posix(),
                "final_sensitivity_level": review["human_final_label"],
                "sensitivity_level": review["human_final_label"],
                "source_dataset": source["source_dataset"],
                "source_label": source["source_label"],
                "source_label_provenance": source["source_label_provenance"],
                "split": "train",
                "internal_split": previous["internal_split"] if previous else assignments[h],
                "annotation_source": "human_review",
                "annotation_file": REVIEW.as_posix(),
                "annotation_row": review["annotation_row"],
                "annotator": review["annotator"],
                "annotated_at": review["annotated_at"],
                "review_status": review["status"],
                "human_final_label": review["human_final_label"],
                "human_initial_label": review["human_initial_label"],
                "model_prediction": review["model_prediction"],
                "model_version": review["model_version"],
                "provenance_json": review["provenance_json"],
                "source_paths_json": source["source_paths_json"],
                "split_seed": "42",
                "split_strategy": "preserved_v1_internal_split"
                if previous
                else "seed42_largest_remainder_20pct_validation_source_dataset_x_human_final_label",
            }
        )
        result[h] = row
    rows = [result[h] for h in sorted(result)]
    identity_checks(rows, "V2 manifest")

    historical = unique_hashes(read_csv(root / HISTORICAL), "historical review")
    reference = unique_hashes(read_csv(root / REFERENCE), "Giovanni reference")
    require(reference.keys() <= reviewed.keys(), "Reference contains additional review images")
    require(
        all(
            row["human_final_label"] == reviewed[h]["human_final_label"]
            for h, row in reference.items()
        ),
        "Giovanni reference differs from canonical source",
    )
    superseded = []
    for h, row in historical.items():
        require(h in reviewed, f"Historical review absent from canonical review: {h}")
        require(
            row["annotated_at"] < reviewed[h]["annotated_at"],
            f"Historical record is not older: {h}",
        )
        if row["human_final_label"] != reviewed[h]["human_final_label"]:
            superseded.append(
                {
                    "content_hash": h,
                    "historical_annotation_file": HISTORICAL.as_posix(),
                    "historical_human_final_label": row["human_final_label"],
                    "historical_annotated_at": row["annotated_at"],
                    "canonical_annotation_file": REVIEW.as_posix(),
                    "canonical_human_final_label": reviewed[h]["human_final_label"],
                    "canonical_annotated_at": reviewed[h]["annotated_at"],
                    "resolution": "superseded_historical_label_user_confirmed_canonical_review_sara",
                }
            )
    inputs = [PACKAGE / name for name in package_audit["csv_sha256"]]
    inputs += [
        PACKAGE / "package_audit.json",
        MASTER,
        REVIEW,
        HISTORICAL,
        REFERENCE,
        Path("annotation_data/splits/automatic_annotator_train.csv"),
        Path("annotation_data/splits/automatic_annotator_test.csv"),
        NOTEBOOK,
        CHECKPOINT,
        PREVIOUS_CONFIG,
    ]
    audit = {
        "manifest": MANIFEST.as_posix(),
        "seed": 42,
        "target_column": "final_sensitivity_level",
        "class_order": list(CLASSES),
        "canonical_review": REVIEW.as_posix(),
        "reviewed_records_before_deduplication": len(reviews),
        "unique_reviewed_images": len(reviewed),
        "previous_train_validation_samples": len(old),
        "previous_effective_train": 449,
        "previous_validation": 112,
        "new_reviewed_added": len(reviewed) - len(replacements),
        "reviewed_replacements": replacements,
        "reviewed_replacement_count": len(replacements),
        "identical_canonical_review_duplicates_collapsed": identical_review_duplicates,
        "superseded_historical_label_count": len(superseded),
        "historical_conflict_report_status": "superseded_by_explicit_user_canonical_source_decision",
        "unresolved_conflicting_labels": [],
        "fixed_test": {"path": (PACKAGE / "test.csv").as_posix(), **distributions(test_rows)},
        "input_sha256": {path.as_posix(): sha256(root / path) for path in inputs},
        "overall": distributions(rows),
        "by_internal_split": {
            part: distributions([row for row in rows if row["internal_split"] == part])
            for part in ("effective_train", "validation")
        },
        "reviewed_by_internal_split": {
            part: distributions(
                [
                    row
                    for row in rows
                    if row["annotation_source"] == "human_review" and row["internal_split"] == part
                ]
            )
            for part in ("effective_train", "validation")
        },
    }
    return rows, audit, superseded


def validate_images(rows: list[dict], root: Path = ROOT) -> dict:
    """Keep unavailable images in metadata; report byte/decoding errors separately."""
    identity_checks(rows, "manifest validation")
    unavailable, mismatches, unreadable = [], [], []
    for row in rows:
        require(row["final_sensitivity_level"] in CLASSES, "Missing/invalid sensitivity label")
        require(
            row["sensitivity_level"] == row["final_sensitivity_level"], "Target aliases disagree"
        )
        require(
            row["internal_split"] in {"effective_train", "validation"}, "Invalid internal split"
        )
        relative = Path(row["image_path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe image_path")
        old_prefix = Path("data/kaggle/sensitivity_resnet50_final")
        if relative.is_relative_to(old_prefix):
            relative = PACKAGE / relative.relative_to(old_prefix)
        path = root / relative
        if not path.is_file():
            unavailable.append(row["image_path"])
            continue
        if sha256(path) != row["content_hash"]:
            mismatches.append(row["image_path"])
        try:
            with Image.open(path) as image:
                ImageOps.exif_transpose(image).convert("RGB").load()
        except (OSError, ValueError):
            unreadable.append(row["image_path"])
    return {
        "total_rows": len(rows),
        "unique_images": len({r["content_hash"] for r in rows}),
        "duplicate_image_ids": [],
        "duplicate_hashes": [],
        "duplicate_canonical_paths": [],
        "missing_image_paths": [],
        "missing_labels": [],
        "invalid_labels": [],
        "conflicting_labels": [],
        "unavailable_local_images": unavailable,
        "image_hash_mismatches": mismatches,
        "unreadable_images": unreadable,
    }


def validate_saved(root: Path = ROOT) -> tuple[list[dict], dict]:
    rows, expected, _ = build_rows(root)
    saved = read_csv(root / MANIFEST)
    # The frozen CSV records original paths. Compare its content after translating
    # only relocatable path fields; do not rewrite provenance or stored checksums.
    old_prefixes = {
        "model_autoannotator/inputs/sensitivity_resnet50_final/": "data/kaggle/sensitivity_resnet50_final/",
        "annotation_data/human-annotation/master/reviewed/": "annotations/master/",
        "annotation_data/model-autoannotation/human_revision/": "annotations/model_annotation/",
        "annotation_data/splits/": "annotations/splits/",
        "model_autoannotator/notebooks/experiments/final output/": "notebooks/final output/",
        "model_autoannotator/notebooks/experiments/": "notebooks/",
    }
    def historical(value: str) -> str:
        for current, original in old_prefixes.items():
            if value.startswith(current):
                return original + value[len(current):]
        return value
    require(len(saved) == len(rows), "V2 manifest row count changed")
    for actual, reconstructed in zip(saved, rows):
        expected_row = dict(reconstructed)
        for field in ("image_path", "annotation_file"):
            expected_row[field] = historical(expected_row[field])
        require(actual == expected_row, "Saved V2 manifest differs from canonical source reconstruction")
    audit = json.loads((root / AUDIT).read_text())
    require(sha256(root / MANIFEST) == audit["manifest_sha256"], "V2 manifest checksum changed")
    checked = {historical(path): digest for path, digest in expected["input_sha256"].items()}
    require(audit["input_sha256"] == checked, "V2 inputs changed after build")
    return saved, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        rows, audit = validate_saved()
        audit["validation"] = validate_images(rows)
    else:
        for path in (MANIFEST, AUDIT, SUPERSEDED):
            require(not (ROOT / path).exists(), f"Refusing to overwrite: {path}")
        rows, audit, superseded = build_rows()
        audit["validation"] = validate_images(rows)
        require(not audit["validation"]["image_hash_mismatches"], "Image SHA-256 mismatch")
        payload = csv_bytes(rows)
        audit["manifest_sha256"] = hashlib.sha256(payload).hexdigest()
        # Exclusive creation protects every old/generated artifact from replacement.
        with (ROOT / MANIFEST).open("xb") as stream:
            stream.write(payload)
        with (ROOT / SUPERSEDED).open("xb") as stream:
            stream.write(
                csv_bytes(
                    superseded,
                    list(superseded[0]) if superseded else ["content_hash", "resolution"],
                )
            )
        with (ROOT / AUDIT).open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
