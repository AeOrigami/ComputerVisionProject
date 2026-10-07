"""Build disjoint, ZIP-importable model-review batches from model predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS = ROOT / "model_autoannotator/notebooks/experiments/final output/automatic_sensitivity_predictions.csv"
DEFAULT_HUMAN = ROOT / "annotation_data/human-annotation/master/reviewed/human_annotations_master_final.csv"
DEFAULT_IMAGES = ROOT / "model_autoannotator/inputs/sensitivity_resnet50_final"
DEFAULT_METRICS = ROOT / "model_autoannotator/notebooks/experiments/final output/test_metrics.json"
OUTPUT_PARENT = ROOT / "manual_annotator/packages/outgoing"
CLASSES = {"low", "medium", "high"}
HASH = re.compile(r"[0-9a-f]{64}\Z")
ASSIGNMENT_FIELDS = (
    "image_id",
    "content_hash",
    "annotator",
    "batch_id",
    "selection_type",
    "source_dataset",
    "source_label",
    "model_prediction",
    "confidence",
    "prob_low",
    "prob_medium",
    "prob_high",
    "model_version",
    "image_path",
    "source_image_path",
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
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        require(bool(reader.fieldnames), f"CSV has no header: {path}")
        rows = list(reader)
    require(all(None not in row for row in rows), f"Malformed CSV: {path}")
    return rows


def csv_bytes(rows: list[dict[str, str]], fields: tuple[str, ...]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def confidence_band(confidence: float) -> str:
    if confidence < 0.5:
        return "<0.50"
    if confidence < 0.7:
        return "0.50-0.70"
    if confidence < 0.9:
        return "0.70-0.90"
    return ">=0.90"


def stratum(row: dict) -> tuple[str, str, str, str]:
    return (
        row["source_dataset"],
        row["source_label"],
        row["predicted_sensitivity_level"],
        confidence_band(row["confidence_value"]),
    )


def proportional_sample(rows: list[dict], count: int, rng: random.Random) -> list[dict]:
    """Largest-remainder allocation across the four requested metadata dimensions."""
    require(count <= len(rows), "Not enough candidates for representative sample")
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[stratum(row)].append(row)
    target = {key: count * len(group) / len(rows) for key, group in groups.items()}
    quotas = {key: math.floor(value) for key, value in target.items()}
    remaining = count - sum(quotas.values())
    for key in sorted(groups, key=lambda item: (-(target[item] - quotas[item]), item)):
        if remaining == 0:
            break
        if quotas[key] < len(groups[key]):
            quotas[key] += 1
            remaining -= 1
    require(remaining == 0, "Could not allocate representative quota")
    selected = []
    for key in sorted(groups):
        group = groups[key][:]
        rng.shuffle(group)
        selected.extend(group[: quotas[key]])
    require(len(selected) == count, "Representative sample size mismatch")
    return selected


def distribute(
    rows: list[dict], people: list[str], each: int, rng: random.Random, *, grouped: bool
) -> dict[str, list[dict]]:
    """Round robin keeps total sizes exact and strata close across annotators."""
    require(len(rows) == len(people) * each, "Assignment size mismatch")
    if grouped:
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for row in rows:
            groups[stratum(row)].append(row)
        ordered = []
        for key in sorted(groups):
            group = groups[key][:]
            rng.shuffle(group)
            ordered.extend(group)
    else:
        ordered = rows[:]
    result = {person: [] for person in people}
    for index, row in enumerate(ordered):
        result[people[index % len(people)]].append(row)
    require(all(len(value) == each for value in result.values()), "Uneven batch assignment")
    return result


def model_score(row: dict, weak_datasets: dict[str, float]) -> float:
    probabilities = sorted(row["probability_values"], reverse=True)
    margin = probabilities[0] - probabilities[1]
    return (
        2.0 * (row["predicted_sensitivity_level"] == "high")
        + (1.0 - row["confidence_value"])
        + (1.0 - margin)
        + 0.25 * (1.0 - weak_datasets.get(row["source_dataset"], 1.0))
    )


def resolved_image(row: dict, image_root: Path) -> Path:
    relative = Path(row["image_path"])
    require(
        not relative.is_absolute()
        and len(relative.parts) == 2
        and relative.parts[0] == "images"
        and relative.stem == row["content_hash"],
        f"Invalid portable image path for {row['content_hash']}: {relative}",
    )
    path = image_root / relative
    require(path.is_file(), f"Missing selected image {row['content_hash']}: {path}")
    require(sha256(path) == row["content_hash"], f"Image SHA-256 mismatch: {path}")
    with Image.open(path) as image:
        image.verify()
    return path


def load_predictions(path: Path, human_hashes: set[str], excluded: set[str]) -> list[dict]:
    rows = read_csv(path)
    required = {
        "content_hash",
        "image_path",
        "source_dataset",
        "source_label",
        "predicted_sensitivity_level",
        "confidence",
        "prob_low",
        "prob_medium",
        "prob_high",
        "model_version",
    }
    require(
        rows and required <= rows[0].keys(),
        f"Missing prediction fields: {required - rows[0].keys() if rows else required}",
    )
    ids = [row["content_hash"] for row in rows]
    require(all(HASH.fullmatch(value) for value in ids), "Invalid prediction content_hash")
    require(len(ids) == len(set(ids)), "Duplicate content_hash in automatic predictions")
    require(not set(ids) & human_hashes, "Automatic predictions include existing human labels")
    for row in rows:
        require(row["predicted_sensitivity_level"] in CLASSES, "Invalid model prediction")
        require(row["source_label"] in {"real", "fake"}, "Invalid source label")
        require(bool(row["source_dataset"] and row["model_version"]), "Missing provenance")
        try:
            probabilities = tuple(
                float(row[f"prob_{level}"]) for level in ("low", "medium", "high")
            )
            confidence = float(row["confidence"])
        except ValueError as exc:
            raise ValueError(f"Invalid probabilities: {row['content_hash']}") from exc
        require(all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities), "Invalid probability")
        require(
            math.isclose(sum(probabilities), 1, abs_tol=1e-5), "Probabilities do not sum to one"
        )
        require(math.isclose(confidence, max(probabilities), abs_tol=1e-5), "Confidence mismatch")
        require(
            row["predicted_sensitivity_level"]
            == ("low", "medium", "high")[probabilities.index(max(probabilities))],
            f"Predicted class/probability mismatch: {row['content_hash']}",
        )
        row["confidence_value"] = confidence
        row["probability_values"] = probabilities
    return [row for row in rows if row["content_hash"] not in excluded]


def build(args: argparse.Namespace) -> dict:
    people = list(args.annotators)
    require(people and len(people) == len(set(people)), "Annotator names must be unique")
    require(
        all(name and re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in people),
        "Annotator names must use letters, digits, underscore or hyphen",
    )
    require(
        args.batch_size > 0 and args.batch_id and re.fullmatch(r"[A-Za-z0-9_-]+", args.batch_id),
        "Invalid batch size or batch ID",
    )
    output = args.output_dir or OUTPUT_PARENT / f"model_review_{args.batch_id}"
    output = output.resolve()
    require(not output.exists(), f"Output already exists; refusing to overwrite: {output}")
    human_rows = read_csv(args.human_master)
    require(human_rows and "content_hash" in human_rows[0], "Human master lacks content_hash")
    human_hashes = {row["content_hash"] for row in human_rows}
    require(len(human_rows) == len(human_hashes) == 701, "Expected 701 unique human hashes")
    require(all(HASH.fullmatch(value) for value in human_hashes), "Invalid human hash")
    excluded = set()
    exclude_checksums = {}
    for path in args.exclude:
        rows = read_csv(path)
        require(
            rows and ({"content_hash", "image_id"} & rows[0].keys()),
            f"Exclusion CSV needs content_hash or image_id: {path}",
        )
        column = "content_hash" if "content_hash" in rows[0] else "image_id"
        ids = {row[column] for row in rows}
        require(all(HASH.fullmatch(value) for value in ids), f"Invalid exclusion ID: {path}")
        excluded |= ids
        exclude_checksums[str(path)] = sha256(path)
    candidates = load_predictions(args.predictions, human_hashes, excluded)
    automatic_ids = {row["content_hash"] for row in read_csv(args.predictions)}
    expected_total = args.batch_size * len(people)
    require(
        len(candidates) >= expected_total,
        f"Only {len(candidates)} eligible model predictions; need {expected_total}",
    )
    weak_datasets = {}
    if args.test_metrics and args.test_metrics.is_file():
        metrics = json.loads(args.test_metrics.read_text(encoding="utf-8"))
        weak_datasets = {
            name: float(values["macro_f1"])
            for name, values in metrics.get("by_source_dataset", {}).items()
        }
    rng = random.Random(args.seed)
    representative_each = round(args.batch_size * 0.60)
    targeted_each = args.batch_size - representative_each
    representative = proportional_sample(candidates, representative_each * len(people), rng)
    representative_ids = {row["content_hash"] for row in representative}
    remainder = [row for row in candidates if row["content_hash"] not in representative_ids]
    rng.shuffle(remainder)  # Stable random tie-breaking before sorting by target priority.
    targeted = sorted(remainder, key=lambda row: model_score(row, weak_datasets), reverse=True)[
        : targeted_each * len(people)
    ]
    require(len(targeted) == targeted_each * len(people), "Targeted sample size mismatch")
    representative_parts = distribute(
        representative, people, representative_each, rng, grouped=True
    )
    targeted_parts = distribute(targeted, people, targeted_each, rng, grouped=False)
    selected_ids = {row["content_hash"] for row in representative + targeted}
    require(not selected_ids & human_hashes, "Review selection overlaps human-labelled images")
    require(selected_ids <= automatic_ids, "Review selection contains non-model images")
    require(len(selected_ids) == expected_total, "Review selection has duplicates or wrong size")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".model_review_{args.batch_id}_", dir=output.parent))
    try:
        assignment = []
        per_batch = {}
        for person in people:
            batch_name = f"{args.batch_id}_{person}"
            chosen = [(row, "representative") for row in representative_parts[person]]
            chosen += [(row, "targeted") for row in targeted_parts[person]]
            rng.shuffle(chosen)
            metadata = {}
            zip_path = staging / f"{batch_name}.zip"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
                for row, selection_type in chosen:
                    source = resolved_image(row, args.image_root)
                    archive_name = f"images/{row['content_hash']}{source.suffix.lower()}"
                    archive.write(source, archive_name)
                    provenance = {
                        "review_type": "model_review",
                        "image_id": row["content_hash"],
                        "content_hash": row["content_hash"],
                        "image_path": archive_name,
                        "source_dataset": row["source_dataset"],
                        "source_label": row["source_label"],
                        "normalized_label": row["source_label"],
                        "label_source": row.get("source_label_provenance") or "prediction_export",
                        "model_prediction": row["predicted_sensitivity_level"],
                        "confidence": row["confidence"],
                        "prob_low": row["prob_low"],
                        "prob_medium": row["prob_medium"],
                        "prob_high": row["prob_high"],
                        "model_version": row["model_version"],
                        "batch_id": batch_name,
                        "assigned_annotator": person,
                        "selection_type": selection_type,
                        "source_image_path": row["image_path"],
                        "source_paths_json": row.get("source_paths_json", ""),
                    }
                    metadata[archive_name] = provenance
                    assignment.append(
                        {
                            "image_id": row["content_hash"],
                            "content_hash": row["content_hash"],
                            "annotator": person,
                            "batch_id": batch_name,
                            "selection_type": selection_type,
                            "source_dataset": row["source_dataset"],
                            "source_label": row["source_label"],
                            "model_prediction": row["predicted_sensitivity_level"],
                            "confidence": row["confidence"],
                            "prob_low": row["prob_low"],
                            "prob_medium": row["prob_medium"],
                            "prob_high": row["prob_high"],
                            "model_version": row["model_version"],
                            "image_path": archive_name,
                            "source_image_path": row["image_path"],
                        }
                    )
                archive.writestr(
                    "metadata.json", json.dumps(metadata, ensure_ascii=False, sort_keys=True)
                )
            require(
                zip_path.stat().st_size <= 512 * 1024 * 1024,
                f"{batch_name}.zip exceeds the annotator's 512 MiB batch limit",
            )
            per_batch[batch_name] = {
                "count": len(chosen),
                "selection_type": dict(Counter(kind for _, kind in chosen)),
                "model_prediction": dict(
                    Counter(row["predicted_sensitivity_level"] for row, _ in chosen)
                ),
                "source_dataset": dict(Counter(row["source_dataset"] for row, _ in chosen)),
                "zip_sha256": sha256(zip_path),
                "zip_bytes": zip_path.stat().st_size,
            }
        require(len(assignment) == expected_total, "Assignment count mismatch")
        require(
            len({row["content_hash"] for row in assignment}) == expected_total,
            "Annotator batches overlap",
        )
        assignment_bytes = csv_bytes(assignment, ASSIGNMENT_FIELDS)
        (staging / "review_assignment.csv").write_bytes(assignment_bytes)
        audit = {
            "batch_id": args.batch_id,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "annotators": people,
            "candidate_predictions_after_exclusion": len(candidates),
            "existing_human_labels": len(human_hashes),
            "excluded_previous_ids": len(excluded),
            "selected_total": expected_total,
            "representative_total": len(representative),
            "targeted_total": len(targeted),
            "checks": {
                "review_ids_intersect_existing_human_ids": False,
                "selected_review_ids_subset_automatic_prediction_ids": True,
                "selected_review_ids_equal_expected_total": True,
                "annotator_batches_disjoint": True,
                "all_selected_images_verified_by_sha256": True,
            },
            "per_batch": per_batch,
            "sampling": {
                "representative": "largest-remainder strata: dataset × real/fake × prediction × confidence band",
                "targeted": "descending 2*high + (1-confidence) + (1-top-two margin) + 0.25*(1-dataset test macro-F1)",
                "test_diagnostic_macro_f1": weak_datasets,
            },
            "input_sha256": {
                "automatic_sensitivity_predictions.csv": sha256(args.predictions),
                "human_annotations_master_final.csv": sha256(args.human_master),
                **(
                    {"test_metrics.json": sha256(args.test_metrics)}
                    if args.test_metrics and args.test_metrics.is_file()
                    else {}
                ),
                "excluded_csv": exclude_checksums,
            },
            "assignment_sha256": hashlib.sha256(assignment_bytes).hexdigest(),
        }
        (staging / "review_audit.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        require(not output.exists(), f"Output appeared during build: {output}")
        staging.rename(output)
        return audit
    except BaseException:
        shutil.rmtree(staging)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--human-master", type=Path, default=DEFAULT_HUMAN)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--test-metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--annotators", nargs="+", default=["sara", "lorenzo", "giovanni"])
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--batch-id", default="batch_01")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = build(args)
    print(
        json.dumps(
            {
                "output": str(args.output_dir or OUTPUT_PARENT / f"model_review_{args.batch_id}"),
                "selected_total": result["selected_total"],
                "checks": result["checks"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
