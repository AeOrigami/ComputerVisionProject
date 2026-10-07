"""Auditable booster preflight and feedback analysis; no split generation or training."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

from model_autoannotator.script.build_human_review_retraining_v2_manifest import (
    AUDIT,
    CHECKPOINT,
    CLASSES,
    MANIFEST,
    MASTER,
    PACKAGE,
    PREVIOUS_CONFIG,
    REVIEW,
    ROOT,
    distributions,
    identity_checks,
    read_csv,
    require,
    sha256,
    unique_hashes,
    validate_images,
)


def historical_audit_key(relative: Path) -> str:
    """Map relocated inputs to keys in the immutable V2 manifest audit."""
    value = relative.as_posix()
    prefixes = {
        "model_autoannotator/inputs/sensitivity_resnet50_final/": "data/kaggle/sensitivity_resnet50_final/",
        "annotation_data/model-autoannotation/human_revision/": "annotations/model_annotation/",
        "annotation_data/human-annotation/master/reviewed/": "annotations/master/",
        "model_autoannotator/notebooks/experiments/final output/": "notebooks/final output/",
    }
    for current, original in prefixes.items():
        if value.startswith(current):
            return original + value[len(current):]
    return value


BOOSTED_VERSION = "resnet50_layer4_fc_human_review_corrective_booster_v2"
BOOSTED_CHECKPOINT = "best_model_boosted.pt"
NOTEBOOK = Path("model_autoannotator/notebooks/experiments/sensitivity_resnet50_human_review_v2.ipynb")
INPUTS = (
    MANIFEST,
    AUDIT,
    REVIEW,
    MASTER,
    PREVIOUS_CONFIG,
    CHECKPOINT,
    PACKAGE / "train.csv",
    PACKAGE / "test.csv",
)


def corrective_fields(row: dict) -> dict:
    reviewed = row["annotation_source"] == "human_review"
    if reviewed:
        require(row["human_final_label"] in CLASSES, "Invalid human_final_label")
        require(row["model_prediction"] in CLASSES, "Invalid model_prediction")
        require(
            row["human_final_label"] == row["final_sensitivity_level"],
            "Human final target mismatch",
        )
    error = reviewed and row["model_prediction"] != row["human_final_label"]
    category = (
        "reviewed_v1_error" if error else "reviewed_v1_correct" if reviewed else "original_v1"
    )
    return {
        "was_v1_error": bool(error),
        "training_category": category,
        "sample_weight": 2.0 if error else 1.0,
    }


def review_error_analysis(rows: list[dict]) -> dict:
    """Rows=true human final, columns=recorded V1 prediction; no fresh inference."""
    reviewed = [row for row in rows if row["annotation_source"] == "human_review"]

    def summarize(group):
        errors = [row for row in group if row["model_prediction"] != row["human_final_label"]]
        n = len(group)
        return {
            "support": n,
            "originally_correct": n - len(errors),
            "originally_incorrect": len(errors),
            "correct_percent": 100 * (n - len(errors)) / n if n else None,
            "incorrect_percent": 100 * len(errors) / n if n else None,
            "errors_by_target_class": {
                label: sum(row["human_final_label"] == label for row in errors) for label in CLASSES
            },
            "confusion_matrix": [
                [
                    sum(
                        row["human_final_label"] == true and row["model_prediction"] == predicted
                        for row in group
                    )
                    for predicted in CLASSES
                ]
                for true in CLASSES
            ],
        }

    return {
        "label": "recorded V1 predictions versus authoritative human final labels",
        "class_order": list(CLASSES),
        "confusion_matrix_axes": "rows=human_final_label, columns=model_prediction",
        **summarize(reviewed),
        "by_source_dataset": {
            source: summarize([r for r in reviewed if r["source_dataset"] == source])
            for source in sorted({r["source_dataset"] for r in reviewed})
        },
        "by_internal_split": {
            part: summarize([r for r in reviewed if r["internal_split"] == part])
            for part in ("effective_train", "validation")
        },
    }


def recovery_analysis(rows: list[dict]) -> dict:
    """After-inference rows must include v2_prediction and the original review fields."""

    def summarize(group):
        errors = [r for r in group if r["model_prediction"] != r["human_final_label"]]
        correct = [r for r in group if r["model_prediction"] == r["human_final_label"]]
        recovered = sum(r["v2_prediction"] == r["human_final_label"] for r in errors)
        return {
            "support": len(group),
            "former_v1_errors": len(errors),
            "errors_recovered": recovered,
            "errors_remaining": len(errors) - recovered,
            "recovery_percent": 100 * recovered / len(errors) if errors else None,
            "previously_correct_now_wrong": sum(
                r["v2_prediction"] != r["human_final_label"] for r in correct
            ),
            "human_final_label_matches": sum(
                r["v2_prediction"] == r["human_final_label"] for r in group
            ),
        }

    require(
        len(rows) == len({r["content_hash"] for r in rows}) == 400,
        "Recovery requires all 400 reviews",
    )
    require(all(r["v2_prediction"] in CLASSES for r in rows), "Invalid V2 recovery predictions")
    return {
        "label": "corrective-set recovery / training-feedback analysis",
        "interpretation": "Not unbiased test performance: 320 examples were used for gradients; 80 for validation selection.",
        **summarize(rows),
        "by_internal_split": {
            part: summarize([r for r in rows if r["internal_split"] == part])
            for part in ("effective_train", "validation")
        },
        "by_source_dataset": {
            source: summarize([r for r in rows if r["source_dataset"] == source])
            for source in sorted({r["source_dataset"] for r in rows})
        },
        "by_target_class": {
            label: summarize([r for r in rows if r["human_final_label"] == label])
            for label in CLASSES
        },
    }


def metric_comparison(v1: dict, v2: dict) -> dict:
    """Positive delta means V2 is higher; retain full per-class and source metrics."""

    def compare(a, b):
        require(a["support"] == b["support"], "Evaluation support mismatch")
        result = {
            name: {"v1": a[name], "boosted_v2": b[name], "delta": b[name] - a[name]}
            for name in ("accuracy", "macro_f1", "balanced_accuracy")
        }
        result["support"] = a["support"]
        result["per_class"] = {
            label: {
                name: {
                    "v1": a["per_class"][label][name],
                    "boosted_v2": b["per_class"][label][name],
                    "delta": b["per_class"][label][name] - a["per_class"][label][name],
                }
                for name in ("precision", "recall", "f1", "support")
            }
            for label in CLASSES
        }
        result["confusion_matrices"] = {
            "v1": a["confusion_matrix"],
            "boosted_v2": b["confusion_matrix"],
        }
        return result

    require(
        v1["class_order"] == v2["class_order"] == list(CLASSES), "Evaluation class order mismatch"
    )
    require(v1["fixed_test_csv_sha256"] == v2["fixed_test_csv_sha256"], "Evaluation test mismatch")
    require(
        set(v1["by_source_dataset"]) == set(v2["by_source_dataset"]), "Evaluation source mismatch"
    )
    return {
        "label": "V1 versus boosted V2 on the frozen 140-image generalization test",
        "fixed_test_csv_sha256": v1["fixed_test_csv_sha256"],
        "class_order": list(CLASSES),
        "v1_checkpoint_sha256": v1["checkpoint_sha256"],
        "boosted_v2_checkpoint_sha256": v2["checkpoint_sha256"],
        "overall": compare(v1, v2),
        "by_source_dataset": {
            source: compare(v1["by_source_dataset"][source], v2["by_source_dataset"][source])
            for source in sorted(v1["by_source_dataset"])
        },
    }


def preflight(
    root: Path = ROOT, *, verify_images: bool = True
) -> tuple[list[dict], list[dict], dict]:
    """Read the locked manifest directly; never call a splitter or build a manifest."""
    root = Path(root)
    audit = json.loads((root / AUDIT).read_text())
    require(
        sha256(root / MANIFEST) == audit["manifest_sha256"], "Canonical manifest checksum changed"
    )
    for relative in INPUTS:
        require((root / relative).is_file(), f"Missing input: {relative}")
        if relative not in (MANIFEST, AUDIT):
            require(
                sha256(root / relative) == audit["input_sha256"][historical_audit_key(relative)],
                f"Audited input changed: {relative}",
            )
    rows = read_csv(root / MANIFEST)
    old = unique_hashes(read_csv(root / PACKAGE / "train.csv"), "V1 train/validation")
    tests = read_csv(root / PACKAGE / "test.csv")
    test_by_hash = unique_hashes(tests, "fixed test")
    canonical = unique_hashes(read_csv(root / REVIEW), "authoritative reviewed CSV")
    master = unique_hashes(read_csv(root / MASTER), "V1 human master")
    config = json.loads((root / PREVIOUS_CONFIG).read_text())
    require(
        tuple(config["class_order"]) == CLASSES and config["seed"] == 42, "V1 class/seed mismatch"
    )
    identity_checks(rows, "booster manifest")
    by_hash = unique_hashes(rows, "booster manifest")
    require(
        len(rows) == 961 and len(old) == 561 and len(tests) == 140 and len(canonical) == 400,
        "Unexpected input counts",
    )
    require(not set(by_hash) & set(test_by_hash), "Train/validation/test hash overlap")
    require(
        not {r["image_id"] for r in rows} & {master[h]["image_id"] for h in test_by_hash},
        "Train/validation/test image ID overlap",
    )
    require(not set(old) & set(canonical), "Unexpected original/review overlap")
    require(set(by_hash) == set(old) | set(canonical), "Manifest membership changed")
    for h, row in by_hash.items():
        require(
            row["final_sensitivity_level"] in CLASSES
            and row["sensitivity_level"] == row["final_sensitivity_level"],
            f"Missing/invalid target: {h}",
        )
        require(
            row["split"] == "train" and row["internal_split"] in {"effective_train", "validation"},
            f"Invalid assignment: {h}",
        )
        require(
            row["image_path"] in {
                (Path("data/kaggle/sensitivity_resnet50_final") / row["image_filename"]).as_posix(),
                (PACKAGE / row["image_filename"]).as_posix(),
            },
            f"Image path mismatch: {h}",
        )
        row["image_path"] = (PACKAGE / row["image_filename"]).as_posix()
        image_name = Path(row["image_filename"])
        require(
            not image_name.is_absolute()
            and len(image_name.parts) == 2
            and image_name.parts[0] == "images"
            and image_name.stem == h,
            f"Unsafe image filename: {h}",
        )
        if h in old:
            require(
                row["internal_split"] == old[h]["internal_split"], f"V1 assignment changed: {h}"
            )
            require(
                row["final_sensitivity_level"] == old[h]["final_sensitivity_level"],
                f"V1 target changed: {h}",
            )
            require(
                row["annotation_source"] != "human_review", f"V1 replay provenance changed: {h}"
            )
        else:
            review = canonical[h]
            require(row["annotation_source"] == "human_review", f"Review provenance missing: {h}")
            for field in (
                "human_final_label",
                "model_prediction",
                "source_dataset",
                "source_label",
                "image_id",
            ):
                require(row[field] == review[field], f"Canonical review mismatch in {field}: {h}")
            require(
                row["model_version"] == config["model_version"], "Recorded predictions are not V1"
            )
        row.update(corrective_fields(row))
    require(
        Counter(r["internal_split"] for r in rows) == {"effective_train": 769, "validation": 192},
        "Manifest partition counts changed",
    )
    reviewed = [r for r in rows if r["annotation_source"] == "human_review"]
    require(
        Counter(r["internal_split"] for r in reviewed)
        == {"effective_train": 320, "validation": 80},
        "Reviewed partition counts changed",
    )
    # Test rows are used only for integrity here, never for training decisions.
    for row in tests:
        require(
            row["split"] == "test" and row["final_sensitivity_level"] in CLASSES,
            "Invalid fixed test",
        )
        row["image_path"] = (PACKAGE / row["image_filename"]).as_posix()
        row["image_id"] = master[row["content_hash"]]["image_id"]
    identity_checks(rows + tests, "all train/validation/test identities")
    validation = None
    if verify_images:
        validation = validate_images(rows, root)
        for field in ("unavailable_local_images", "image_hash_mismatches", "unreadable_images"):
            require(
                not validation[field], f"Image verification failed: {field}={validation[field]}"
            )
        for row in tests:
            path = root / row["image_path"]
            require(
                path.is_file() and sha256(path) == row["content_hash"],
                f"Test image unavailable/changed: {path}",
            )
            with Image.open(path) as image:
                ImageOps.exif_transpose(image).convert("RGB").load()
    training = [r for r in rows if r["internal_split"] == "effective_train"]
    report = {
        "manifest": MANIFEST.as_posix(),
        "manifest_sha256": audit["manifest_sha256"],
        "input_sha256": {p.as_posix(): sha256(root / p) for p in INPUTS},
        "seed": 42,
        "class_order": list(CLASSES),
        "manifest_rows": len(rows),
        "unique_images": len(by_hash),
        "training": 769,
        "validation": 192,
        "original_replay_training": 449,
        "original_validation": 112,
        "reviewed_training": 320,
        "reviewed_validation": 80,
        "fixed_test": 140,
        "reviewed_training_v1_errors": sum(r["was_v1_error"] for r in training),
        "training_categories": dict(Counter(r["training_category"] for r in training)),
        "sample_weight_counts": dict(Counter(str(r["sample_weight"]) for r in training)),
        "split_distributions": {
            part: distributions([r for r in rows if r["internal_split"] == part])
            for part in ("effective_train", "validation")
        },
        "image_integrity": validation,
        "image_verification_performed": verify_images,
        "review_error_analysis": review_error_analysis(rows),
        "unresolved_annotation_conflicts": 0,
        "historical_differences": "45 superseded labels; historical exports supply no booster targets",
        "split_policy": "read existing manifest; no split generation",
        "checkpoint_load": "not_checked_requires_pytorch",
    }
    return rows, tests, report
