"""Evaluate annotated SensiFake images with DeepfakeBench's Effort detector.

The script imports DeepfakeBench instead of copying its model code. It keeps the
SensiFake manifest as the source of truth for image paths, labels, and sensitivity.
"""

from __future__ import annotations

import argparse
import csv
import json 
import json as simplejson
import os
import sys
from collections import Counter
from pathlib import Path
import re
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LEVELS = ("low", "medium", "high")
SENSITIVITY_WEIGHTS = {"low": 1.0, "medium": 2.0, "high": 3.0}
DEFAULT_ANNOTATIONS = ROOT / "benchmark/data/sensifake-hf/metadata/sensifake_all.csv"
DEFAULT_DEEPFAKEBENCH = ROOT / "external/DeepfakeBench"
DEFAULT_DETECTOR_CONFIG = DEFAULT_DEEPFAKEBENCH / "training/config/detector/effort.yaml"
DEFAULT_EFFORT_MODEL = DEFAULT_DEEPFAKEBENCH / "huggingface/clip-vit-large-patch14"
DEFAULT_WEIGHTS = DEFAULT_DEEPFAKEBENCH / "training/weights/effort_clip_L14_trainOn_sdv14.pth"
DEFAULT_OUTPUT = ROOT / "benchmark/results/deepfakebench-effort-hf"


def repository_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} not found: {path}. "
            "Clone DeepfakeBench and pass paths from that checkout."
        )


def require_directory(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{description} not found: {path}")


def annotated_rows(manifest: Path, root: Path = ROOT) -> list[dict[str, Any]]:
    """Return the complete, evaluable human-annotated population."""
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = []
    for row in rows:
        if (
            row.get("file_status") == "readable"
            and row.get("annotation_join_status") == "linked"
            and row.get("label_verification_status") == "verified"
            and row.get("normalized_label") in ("real", "fake")
            and row.get("sensitivity_level") in LEVELS
        ):
            image_path = (root / row["image_path"]).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"Image referenced by manifest is missing: {image_path}")
            selected.append({
                **row,
                "source_dataset": row.get("source_dataset") or row.get("dataset") or "unknown",
                "resolved_image_path": image_path,
            })
    if not selected:
        raise ValueError("No readable, linked, verified annotated images were found")
    return selected


def automatic_rows(csv_path: Path, root: Path = ROOT) -> list[dict[str, Any]]:
    """Load automatic sensitivity predictions and verified source labels."""
    def resolve_image_path(reference: str) -> Path | None:
        candidates = (
            csv_path.parent.parent / reference,
            root / reference,
        )
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.is_file():
                return resolved
        return None

    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = []
    for row in rows:
        if row.get("real_fake_label_name") in ("real", "fake"):
            sensitivity_level = row.get("final_sensitivity_label")
            image_reference = row.get("image_path") or row.get("source_original_path")
            if not image_reference:
                continue
            image_path = resolve_image_path(image_reference)
            if sensitivity_level in LEVELS and image_path is not None:
                selected.append({
                    "content_hash": row["sha256"],
                    "image_path": image_reference,
                    "source_dataset": row.get("source_dataset") or "unknown",
                    "sensitivity_level": sensitivity_level,
                    "normalized_label": row["real_fake_label_name"],
                    "resolved_image_path": image_path,
                })
            continue
        sensitivity_level = (
            row.get("adjudicated_sensitivity_level")
            or row.get("sensitivity_level")
            or row.get("predicted_sensitivity_level")
        )
        if (
            row.get("source_label") in ("real", "fake")
            and sensitivity_level in LEVELS
        ):
            source_paths = json.loads(row["source_paths_json"])
            if not source_paths:
                continue
            image_path = resolve_image_path(source_paths[0])
            if image_path is None:
                continue
            selected.append({
                "content_hash": row["content_hash"],
                "image_path": source_paths[0],
                "source_dataset": row.get("source_dataset") or "unknown",
                "sensitivity_level": sensitivity_level,
                "normalized_label": row["source_label"],
                "resolved_image_path": image_path,
            })
    if not selected:
        raise ValueError("No valid automatic sensitivity predictions were found")
    return selected


def dataset_name(row: dict[str, Any]) -> str:
    """Return a stable dataset label for old and published manifests."""
    return str(row.get("source_dataset") or row.get("dataset") or "unknown")


def grouped_metrics(
    rows: list[dict[str, Any]], group_key: str
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[group_key]), []).append(row)
    return {
        group: binary_metrics(
            [row["label"] for row in group_rows],
            [row["probability_fake"] for row in group_rows],
        )
        for group, group_rows in sorted(groups.items())
    }


def binary_metrics(labels: list[int], probabilities: list[float]) -> dict[str, Any]:
    """Compute metrics without requiring scikit-learn in the host project."""
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("labels and probabilities must be non-empty and equally sized")
    predicted = [int(probability >= 0.5) for probability in probabilities]
    tn = sum(label == 0 and prediction == 0 for label, prediction in zip(labels, predicted))
    fp = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predicted))
    fn = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predicted))
    tp = sum(label == 1 and prediction == 1 for label, prediction in zip(labels, predicted))
    positive = sum(labels)
    negative = len(labels) - positive
    true_positive_rate = tp / positive if positive else None
    true_negative_rate = tn / negative if negative else None
    auc = None
    if positive and negative:
        ordered = sorted(zip(probabilities, labels), key=lambda item: item[0])
        rank_sum = sum(index for index, (_, label) in enumerate(ordered, 1) if label)
        auc = (rank_sum - positive * (positive + 1) / 2) / (positive * negative)
    return {
        "count": len(labels),
        "real": negative,
        "fake": positive,
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": (
            (true_positive_rate + true_negative_rate) / 2
            if true_positive_rate is not None and true_negative_rate is not None
            else None
        ),
        "precision_fake": tp / (tp + fp) if tp + fp else None,
        "recall_fake": true_positive_rate,
        "f1_fake": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "roc_auc": auc,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def sensitivity_weighted_accuracy(rows: list[dict[str, Any]]) -> float:
    """Score accuracy while weighting high-sensitivity samples more heavily."""
    if not rows:
        raise ValueError("rows must be non-empty")
    total_weight = 0.0
    correct_weight = 0.0
    for row in rows:
        try:
            weight = SENSITIVITY_WEIGHTS[row["sensitivity_level"]]
        except KeyError as error:
            raise ValueError(
                f"Unknown sensitivity level: {row.get('sensitivity_level')!r}"
            ) from error
        total_weight += weight
        correct_weight += weight * (
            int(row["probability_fake"] >= 0.5) == row["label"]
        )
    return correct_weight / total_weight


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def plot_metric_comparison(
    output: Path,
    grouped: dict[str, dict[str, Any]],
    title: str,
    filename: str,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    groups = list(grouped)
    metric_names = (
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("precision_fake", "Fake precision"),
        ("recall_fake", "Fake recall"),
        ("f1_fake", "Fake F1"),
        ("roc_auc", "ROC-AUC"),
        ("sensitivity_weighted_accuracy", "Sensitivity-weighted accuracy"),
    )
    figure, axis = plt.subplots(figsize=(max(8, len(groups) * 1.3), 5))
    positions = np.arange(len(groups))
    width = 0.8 / len(metric_names)
    for index, (key, label) in enumerate(metric_names):
        values = [grouped[group].get(key) or 0 for group in groups]
        axis.bar(positions + index * width, values, width, label=label)
    axis.set(
        xticks=positions + width * (len(metric_names) - 1) / 2,
        xticklabels=groups,
        ylim=(0, 1.05),
        ylabel="Score",
        title=title,
    )
    axis.tick_params(axis="x", labelrotation=35)
    axis.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    figure.tight_layout()
    figure.savefig(output / filename, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_counts(output: Path, grouped: dict[str, dict[str, Any]], title: str, filename: str) -> None:
    import matplotlib.pyplot as plt

    groups = list(grouped)
    real = [grouped[group]["real"] for group in groups]
    fake = [grouped[group]["fake"] for group in groups]
    figure, axis = plt.subplots(figsize=(max(7, len(groups) * 1.2), 4.5))
    positions = list(range(len(groups)))
    axis.bar(positions, real, label="Real", color="#2563eb")
    axis.bar(positions, fake, bottom=real, label="Fake", color="#dc2626")
    axis.set(
        xticks=positions,
        xticklabels=groups,
        ylabel="Images",
        title=title,
    )
    axis.tick_params(axis="x", labelrotation=35)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / filename, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_dataset_confusion_matrices(
    output: Path, grouped: dict[str, dict[str, Any]]
) -> None:
    import matplotlib.pyplot as plt

    for dataset, metrics in grouped.items():
        confusion = metrics["confusion_matrix"]
        matrix = [[confusion["tn"], confusion["fp"]], [confusion["fn"], confusion["tp"]]]
        figure, axis = plt.subplots(figsize=(4.5, 4))
        image = axis.imshow(matrix, cmap="Blues")
        axis.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["real", "fake"],
            yticklabels=["real", "fake"],
            xlabel="Predicted",
            ylabel="Annotated",
            title=f"Confusion matrix: {dataset}",
        )
        for row_index in range(2):
            for column_index in range(2):
                axis.text(
                    column_index,
                    row_index,
                    matrix[row_index][column_index],
                    ha="center",
                    va="center",
                )
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(output / f"confusion_matrix_{safe_name(dataset)}.png", dpi=160)
        plt.close(figure)


def plot_f1_heatmap(
    output: Path, grouped: dict[tuple[str, str], dict[str, Any]]
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    datasets = sorted({dataset for dataset, _ in grouped})
    levels = list(LEVELS)
    values = np.full((len(datasets), len(levels)), np.nan)
    for row_index, dataset in enumerate(datasets):
        for column_index, level in enumerate(levels):
            metrics = grouped.get((dataset, level))
            if metrics and metrics["f1_fake"] is not None:
                values[row_index, column_index] = metrics["f1_fake"]
    figure, axis = plt.subplots(figsize=(6, max(3.5, len(datasets) * 0.55)))
    image = axis.imshow(values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axis.set(
        xticks=range(len(levels)),
        yticks=range(len(datasets)),
        xticklabels=levels,
        yticklabels=datasets,
        xlabel="Sensitivity level",
        title="Fake F1 by source dataset and sensitivity",
    )
    for row_index in range(len(datasets)):
        for column_index in range(len(levels)):
            value = values[row_index, column_index]
            if not np.isnan(value):
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                )
    figure.colorbar(image, ax=axis, label="Fake F1")
    figure.tight_layout()
    figure.savefig(output / "dataset_sensitivity_f1_heatmap.png", dpi=160)
    plt.close(figure)


def write_group_results(
    output: Path,
    group_name: str,
    grouped_rows: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    metrics = {
        group: binary_metrics(
            [row["label"] for row in rows],
            [row["probability_fake"] for row in rows],
        )
        for group, rows in sorted(grouped_rows.items())
    }
    for group, rows in grouped_rows.items():
        metrics[group]["sensitivity_weighted_accuracy"] = sensitivity_weighted_accuracy(rows)
    metrics_dir = output / group_name
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for group, rows in sorted(grouped_rows.items()):
        group_dir = metrics_dir / safe_name(group)
        group_dir.mkdir(parents=True, exist_ok=True)
        write_csv(group_dir / "predictions.csv", rows)
        (group_dir / "metrics.json").write_text(
            json.dumps(metrics[group], indent=2) + "\n", encoding="utf-8"
        )
    (output / f"{group_name}_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    summary_rows = []
    for group, group_metrics in metrics.items():
        summary_rows.append({
            "group": group,
            "count": group_metrics["count"],
            "real": group_metrics["real"],
            "fake": group_metrics["fake"],
            "accuracy": group_metrics["accuracy"],
            "balanced_accuracy": group_metrics["balanced_accuracy"],
            "precision_fake": group_metrics["precision_fake"],
            "recall_fake": group_metrics["recall_fake"],
            "f1_fake": group_metrics["f1_fake"],
            "roc_auc": group_metrics["roc_auc"],
            "sensitivity_weighted_accuracy": group_metrics[
                "sensitivity_weighted_accuracy"
            ],
        })
    write_csv(output / f"{group_name}_metrics.csv", summary_rows)
    return metrics


def plot_level(output: Path, level: str, metrics: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    level_dir = output / level
    level_dir.mkdir(parents=True, exist_ok=True)
    confusion = metrics["confusion_matrix"]
    figure, axis = plt.subplots(figsize=(5, 4))
    matrix = [[confusion["tn"], confusion["fp"]], [confusion["fn"], confusion["tp"]]]
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["real", "fake"],
             yticklabels=["real", "fake"], xlabel="Predicted", ylabel="Annotated",
             title=f"Effort confusion matrix: {level}")
    for row_index in range(2):
        for column_index in range(2):
            axis.text(column_index, row_index, matrix[row_index][column_index], ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(level_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 4))
    for label, color in ((0, "#2563eb"), (1, "#dc2626")):
        values = sorted(row["probability_fake"] for row in rows if row["label"] == label)
        if values:
            axis.plot(range(1, len(values) + 1), values, ".", color=color,
                      label="fake" if label else "real")
    axis.set(xlabel="Samples sorted by fake probability", ylabel="P(fake)", title=f"Effort scores: {level}")
    axis.legend()
    figure.tight_layout()
    figure.savefig(level_dir / "probabilities.png", dpi=160)
    plt.close(figure)


def load_effort_model(args: argparse.Namespace, config: dict[str, Any]):
    """Load Effort and redirect its hardcoded CLIP path to the CLI path."""
    deepfakebench = args.deepfakebench.resolve()
    training = deepfakebench / "training"
    require_directory(training / "detectors", "DeepfakeBench training package")
    require_directory(args.effort_model, "Effort Hugging Face CLIP model directory")
    landmark_model = deepfakebench / "preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat"
    if not landmark_model.is_file():
        raise FileNotFoundError(
            f"DeepfakeBench landmark model not found: {landmark_model}. "
            "Download it from https://github.com/SCLBD/DeepfakeBench/releases/download/"
            "v1.0.0/shape_predictor_81_face_landmarks.dat"
        )
    sys.path.insert(0, str(training))
    previous_cwd = Path.cwd()
    os.chdir(deepfakebench)
    try:
        import torch
        from transformers import CLIPModel

        original_from_pretrained = CLIPModel.from_pretrained

        def redirected_from_pretrained(cls, model_name_or_path, *positional, **keyword):
            return original_from_pretrained(args.effort_model, *positional, **keyword)

        CLIPModel.from_pretrained = classmethod(redirected_from_pretrained)
        from detectors import DETECTOR

        model = DETECTOR["effort"](config)
        checkpoint = torch.load(args.weights, map_location="cpu")
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "model", "net"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break
        checkpoint = {key.removeprefix("module."): value for key, value in checkpoint.items()}
        incompatible = model.load_state_dict(checkpoint, strict=False)
        if len(incompatible.missing_keys) > 4:
            raise RuntimeError(
                "Effort checkpoint did not match the model; missing keys include "
                f"{incompatible.missing_keys[:5]}"
            )
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        return model.to(device).eval(), torch, device
    finally:
        os.chdir(previous_cwd)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    args.annotations = repository_path(args.annotations)
    args.deepfakebench = repository_path(args.deepfakebench)
    args.detector_config = repository_path(args.detector_config)
    args.effort_model = repository_path(args.effort_model)
    args.weights = repository_path(args.weights)
    args.output = repository_path(args.output)
    require_file(args.annotations, "SensiFake annotation CSV")
    require_file(args.detector_config, "DeepfakeBench Effort detector config")
    require_directory(args.effort_model, "Effort Hugging Face CLIP model directory")
    if not args.weights.is_file():
        raise FileNotFoundError(
            f"Effort detector checkpoint not found: {args.weights}. "
            "DeepfakeBench v1.0.1 does not publish an Effort checkpoint; "
            "do not substitute xception_best.pth. Provide a compatible Effort "
            "checkpoint or train Effort first."
        )
    import numpy as np
    import yaml

    with args.annotations.open(encoding="utf-8", newline="") as stream:
        fields = set(csv.DictReader(stream).fieldnames or ())
    rows = automatic_rows(args.annotations) if (
        {"predicted_sensitivity_level", "real_fake_label_name"} & fields
    ) else annotated_rows(args.annotations)
    with args.detector_config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    model, torch, device = load_effort_model(args, config)
    mean = torch.tensor(args.mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(args.std, dtype=torch.float32).view(3, 1, 1)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start:start + args.batch_size]
            images = []
            for row in batch_rows:
                from PIL import Image
                with Image.open(row["resolved_image_path"]) as image:
                    image = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
                    images.append(np.asarray(image, dtype=np.float32) / 255.0)
            tensor = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)
            tensor = ((tensor - mean) / std).to(device)
            output = model({"image": tensor, "label": torch.zeros(len(images), dtype=torch.long, device=device)}, inference=True)
            probabilities = output["prob"].detach().cpu().tolist()
            for row, probability in zip(batch_rows, probabilities):
                predictions.append({"content_hash": row["content_hash"], "image_path": row["image_path"],
                                    "source_dataset": dataset_name(row),
                                    "sensitivity_level": row["sensitivity_level"],
                                    "label": int(row["normalized_label"] == "fake"),
                                    "probability_fake": probability})

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "predictions.csv", predictions)
    (args.output / "predictions.json").write_text(
        json.dumps(predictions, indent=2) + "\n", encoding="utf-8"
    )
    metrics = {
        "all": binary_metrics(
            [r["label"] for r in predictions],
            [r["probability_fake"] for r in predictions],
        )
    }
    metrics["all"]["sensitivity_weighted_accuracy"] = sensitivity_weighted_accuracy(predictions)
    for level in LEVELS:
        level_rows = [row for row in predictions if row["sensitivity_level"] == level]
        if not level_rows:
            raise ValueError(f"No annotated images in sensitivity level {level}")
        metrics[level] = binary_metrics([r["label"] for r in level_rows], [r["probability_fake"] for r in level_rows])
        metrics[level]["sensitivity_weighted_accuracy"] = sensitivity_weighted_accuracy(level_rows)
        write_csv(args.output / level / "predictions.csv", level_rows)
        (args.output / level / "metrics.json").write_text(json.dumps(metrics[level], indent=2) + "\n", encoding="utf-8")
        plot_level(args.output, level, metrics[level], level_rows)
    dataset_rows: dict[str, list[dict[str, Any]]] = {}
    for row in predictions:
        dataset_rows.setdefault(row["source_dataset"], []).append(row)
    metrics["datasets"] = write_group_results(args.output, "datasets", dataset_rows)
    metrics["dataset_counts"] = {
        dataset: {
            "count": dataset_metrics["count"],
            "real": dataset_metrics["real"],
            "fake": dataset_metrics["fake"],
        }
        for dataset, dataset_metrics in metrics["datasets"].items()
    }
    plot_metric_comparison(args.output, metrics["datasets"], "Effort metrics by source dataset", "dataset_metrics.png")
    plot_counts(args.output, metrics["datasets"], "SensiFake images by source dataset", "dataset_class_counts.png")
    plot_dataset_confusion_matrices(args.output, metrics["datasets"])
    dataset_level_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in predictions:
        key = (row["source_dataset"], row["sensitivity_level"])
        dataset_level_rows.setdefault(key, []).append(row)
    dataset_level_metrics = {
        key: binary_metrics(
            [row["label"] for row in rows],
            [row["probability_fake"] for row in rows],
        )
        for key, rows in sorted(dataset_level_rows.items())
    }
    metrics["dataset_sensitivity"] = {
        f"{dataset}::{level}": group_metrics
        for (dataset, level), group_metrics in dataset_level_metrics.items()
    }
    (args.output / "dataset_sensitivity_metrics.json").write_text(
        json.dumps(metrics["dataset_sensitivity"], indent=2) + "\n",
        encoding="utf-8",
    )
    plot_f1_heatmap(args.output, dataset_level_metrics)
    metrics["selection_counts"] = dict(Counter(row["sensitivity_level"] for row in predictions))
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=DEFAULT_ANNOTATIONS,
        help=f"Annotation CSV (default: {DEFAULT_ANNOTATIONS})",
    )
    parser.add_argument(
        "--deepfakebench",
        type=Path,
        default=DEFAULT_DEEPFAKEBENCH,
        help=f"DeepfakeBench checkout (default: {DEFAULT_DEEPFAKEBENCH})",
    )
    parser.add_argument(
        "--detector-config",
        type=Path,
        default=DEFAULT_DETECTOR_CONFIG,
        help=f"Effort detector config (default: {DEFAULT_DETECTOR_CONFIG})",
    )
    parser.add_argument(
        "--effort-model",
        type=Path,
        default=DEFAULT_EFFORT_MODEL,
        help=f"Local CLIP ViT-L/14 directory (default: {DEFAULT_EFFORT_MODEL})",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help=f"Effort checkpoint (default: {DEFAULT_WEIGHTS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mean", type=float, nargs=3, default=(0.48145466, 0.4578275, 0.40821073))
    parser.add_argument("--std", type=float, nargs=3, default=(0.26862954, 0.26130258, 0.27577711))
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), indent=2))