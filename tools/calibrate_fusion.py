#!/usr/bin/env python3
"""Offline fusion calibration from saved validation probabilities; no model inference.

Run from any directory. See tools/README.md for the missing validation export.
Test predictions are opened only if --test is supplied, after validation selection.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# Experiment settings: only EPSILON varies; the accuracy allowance stays fixed.
EPSILON_MIN = 0.000
EPSILON_MAX = 0.500
EPSILON_STEP = 0.005
MAX_ACCURACY_LOSS = 0.03
# Saved AMP/float16 outputs need not be exact complements. Never renormalize them.
PROBABILITY_SUM_ATOL = 0.001
DISAGREEMENT_EXAMPLES_PER_EPSILON = 5
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_NOTEBOOK = PROJECT_ROOT / "SensiFakeProject-v3.ipynb"
OUTPUT_DIR = PROJECT_ROOT / "formula_test_output"
OUTPUT_FOLDERS = (
    PROJECT_ROOT / "SensiFakeProject-output-v2",
    PROJECT_ROOT / "SensiFakeProject-output-v3-validation",
    PROJECT_ROOT / "SensiFakeProject-output",
)
VALIDATION_FILENAMES = (
    "fusion_validation_predictions.csv", "validation_predictions.csv",
    "fusion_validation_predictions.json", "validation_predictions.json",
)
PROBABILITY_COLUMNS = ["p_real", "p_fake", "p_low", "p_medium", "p_high"]
METRIC_COLUMNS = ["accuracy", "macro_f1", "fake_precision", "fake_recall", "fake_f1", "fake_fnr"]


class InputError(ValueError):
    """Missing or unsuitable saved predictions; never fall back to test data."""


def load_project_config(notebook_path):
    """Read literal weights and release label semantics without executing a notebook."""
    notebook = json.loads(Path(notebook_path).read_text(encoding="utf-8"))
    weights, mappings = {}, {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for statement in ast.parse("".join(cell["source"])).body:
            if not isinstance(statement, ast.Assign):
                continue
            for target in statement.targets:
                if not isinstance(target, ast.Name):
                    continue
                if target.id in ("W_L", "W_M", "W_H"):
                    value = float(ast.literal_eval(statement.value))
                    if target.id in weights and weights[target.id] != value:
                        raise InputError(f"Conflicting {target.id} definitions in {notebook_path}.")
                    weights[target.id] = value
                if target.id in ("adn_indices", "csn_indices"):
                    call = statement.value
                    if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                            and call.func.id == "_semantic_indices" and len(call.args) >= 2):
                        mappings[target.id] = ast.literal_eval(call.args[1])
    if set(weights) != {"W_L", "W_M", "W_H"} or set(mappings) != {"adn_indices", "csn_indices"}:
        raise InputError("Notebook must supply the fixed fusion weights and explicit label semantics.")
    if set(mappings["adn_indices"].values()) != {"real", "fake"}:
        raise InputError("Expected the project's binary Real/Fake mapping.")
    if set(mappings["csn_indices"].values()) != {"low", "medium", "high"}:
        raise InputError("Expected the project's Low/Medium/High mapping.")
    if not all(np.isfinite(v) and v >= 0 for v in weights.values()):
        raise InputError("Prudence weights must be finite and nonnegative.")
    return weights, mappings


def _map_labels(values, mapping, column):
    def semantic(value):
        text = str(value).strip().lower()
        if text in mapping.values():
            return text
        try:
            numeric = float(text)
        except ValueError:
            numeric = np.nan
        if np.isfinite(numeric) and numeric.is_integer() and int(numeric) in mapping:
            return mapping[int(numeric)]
        raise InputError(f"Unknown or missing {column} label: {value!r}; use the project mapping.")
    return values.map(semantic)


def _read_predictions(path, phase, mappings):
    path = Path(path).resolve()
    # fusion_predictions.csv is explicitly the complete notebook's TEST export.
    if phase == "validation" and (path.name == "fusion_predictions.csv" or "test" in path.stem.lower()):
        raise InputError(f"Refusing test predictions as validation input: {path}")
    if not path.is_file():
        raise InputError(f"{phase.title()} prediction file not found: {path}")
    if path.suffix.lower() == ".csv":
        try:
            frame = pd.read_csv(path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            raise InputError(f"Invalid prediction CSV: {path}: {exc}") from exc
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload["predictions"] if isinstance(payload, dict) and "predictions" in payload else payload
        if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
            raise InputError(f"{path} must contain per-image JSON row records, not aggregate metrics.")
        frame = pd.DataFrame(records)
    else:
        raise InputError("Prediction inputs must be CSV or JSON row records.")
    missing = [c for c in ["true_real_fake", *PROBABILITY_COLUMNS] if c not in frame]
    if not any(c in frame for c in ("image_id", "image_path")):
        missing.append("image_id or image_path")
    if missing:
        raise InputError(f"{path} is missing per-image quantities: {', '.join(missing)}")
    if frame.empty:
        raise InputError(f"{phase.title()} predictions contain no images.")
    if "split" in frame:
        allowed = {"validation", "val"} if phase == "validation" else {"test"}
        if not frame["split"].astype(str).str.lower().str.strip().isin(allowed).all():
            raise InputError(f"{path} contains rows outside the {phase} split.")
    identifier = "image_path" if "image_path" in frame else "image_id"
    if frame[identifier].isna().any() or frame[identifier].astype(str).str.strip().eq("").any():
        raise InputError(f"Missing {identifier} values in {path}.")
    if frame[identifier].duplicated().any():
        raise InputError(f"Repeated {identifier} values in {path}; expected one row per image.")
    try:
        frame[PROBABILITY_COLUMNS] = frame[PROBABILITY_COLUMNS].apply(pd.to_numeric, errors="raise").astype(float)
    except (ValueError, TypeError) as exc:
        raise InputError("Probabilities must be numeric.") from exc
    probabilities = frame[PROBABILITY_COLUMNS].to_numpy()
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise InputError("Probabilities must be finite values in [0, 1].")
    if not np.allclose(frame[["p_low", "p_medium", "p_high"]].sum(axis=1), 1,
                       rtol=0, atol=PROBABILITY_SUM_ATOL):
        raise InputError("CSN probabilities do not sum approximately to one; check the export.")
    frame = frame.reset_index(drop=True)
    frame["_authenticity"] = _map_labels(frame["true_real_fake"], mappings["adn_indices"], "true_real_fake")
    if "true_sensitivity" in frame:
        frame["_sensitivity"] = _map_labels(frame["true_sensitivity"], mappings["csn_indices"], "true_sensitivity")
    for column in ("training_run_id", "split_fingerprint", "adn_checkpoint", "csn_checkpoint"):
        if column in frame and (frame[column].isna().any() or frame[column].nunique() != 1):
            raise InputError(f"Missing or mixed provenance in {column}.")
    return frame


def load_validation_predictions(path, mappings):
    return _read_predictions(path, "validation", mappings)


def compute_prudence_score(frame, weights):
    return (weights["W_L"] * frame["p_low"].to_numpy()
            + weights["W_M"] * frame["p_medium"].to_numpy()
            + weights["W_H"] * frame["p_high"].to_numpy())


def fusion_from_p_real(p_real, prudence_score, epsilon):
    threshold = 0.5 + epsilon * prudence_score
    return np.where(np.asarray(p_real) > threshold, "real", "fake"), threshold


def fusion_from_p_fake(p_fake, prudence_score, epsilon):
    threshold = 0.5 - epsilon * prudence_score
    return np.where(np.asarray(p_fake) >= threshold, "fake", "real"), threshold


def compute_metrics(y_true, y_pred):
    fake_true = np.asarray(y_true) == "fake"
    fake_pred = np.asarray(y_pred) == "fake"
    recall = recall_score(fake_true, fake_pred, zero_division=0) if fake_true.any() else np.nan
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=["real", "fake"], average="macro", zero_division=0),
        "fake_precision": precision_score(fake_true, fake_pred, zero_division=0),
        "fake_recall": recall,
        "fake_f1": f1_score(fake_true, fake_pred, zero_division=0),
        "fake_fnr": float((fake_true & ~fake_pred).sum() / fake_true.sum()) if fake_true.any() else np.nan,
    }


def analyze_decision_changes(y_true, baseline, fusion):
    y_true, baseline, fusion = map(np.asarray, (y_true, baseline, fusion))
    changed = baseline != fusion
    corrections = int(((baseline != y_true) & (fusion == y_true)).sum())
    regressions = int(((baseline == y_true) & (fusion != y_true)).sum())
    assert corrections + regressions == int(changed.sum())
    transitions = []
    for before, after in (("real", "real"), ("real", "fake"), ("fake", "fake"), ("fake", "real")):
        mask = (baseline == before) & (fusion == after)
        transitions.append({
            "adn_decision": before, "fusion_decision": after, "samples": int(mask.sum()),
            "changed": before != after,
            "corrections": int((mask & (baseline != y_true) & (fusion == y_true)).sum()),
            "regressions": int((mask & (baseline == y_true) & (fusion != y_true)).sum()),
        })
    assert sum(row["samples"] for row in transitions) == len(y_true)
    return {
        "changed_predictions": int(changed.sum()), "corrections": corrections, "regressions": regressions,
        "real_to_fake": transitions[1]["samples"], "fake_to_real": transitions[3]["samples"],
    }, pd.DataFrame(transitions)


def epsilon_grid(minimum=EPSILON_MIN, maximum=EPSILON_MAX, step=EPSILON_STEP):
    low, high, increment = map(lambda v: Decimal(str(v)), (minimum, maximum, step))
    if not all(v.is_finite() for v in (low, high, increment)) or low < 0 or high < low or increment <= 0:
        raise InputError("EPSILON bounds must be finite, nonnegative and ordered, with positive step.")
    count = (high - low) / increment
    if count != count.to_integral_value():
        raise InputError("EPSILON step must divide the configured interval exactly.")
    return np.array([float(low + i * increment) for i in range(int(count) + 1)])


def _sensitivity_rows(frame, predictions, epsilon):
    rows = []
    if "_sensitivity" not in frame:
        return rows
    for level in ("low", "medium", "high"):
        mask = frame["_sensitivity"].to_numpy() == level
        metrics = compute_metrics(frame.loc[mask, "_authenticity"].to_numpy(), predictions[mask]) if mask.any() else {}
        rows.append({"epsilon": epsilon, "sensitivity": level.title(), "samples": int(mask.sum()),
                     **{key: metrics.get(key, np.nan) for key in ("accuracy", "fake_recall", "fake_fnr")}})
    return rows


def run_epsilon_sweep(validation, epsilons, weights):
    """Phase A: this function has no test input, model, optimizer, or inference."""
    truth = validation["_authenticity"].to_numpy()
    rho = compute_prudence_score(validation, weights)
    baseline, _ = fusion_from_p_real(validation["p_real"], rho, 0.0)
    baseline_metrics = compute_metrics(truth, baseline)
    if not np.isfinite(baseline_metrics["fake_recall"]):
        raise InputError("Validation has no true Fake images; the required recall constraint is undefined.")
    rows, groups, examples = [], [], []
    for epsilon in epsilons:
        predicted, real_threshold = fusion_from_p_real(validation["p_real"], rho, epsilon)
        alternate, fake_threshold = fusion_from_p_fake(validation["p_fake"], rho, epsilon)
        metrics = compute_metrics(truth, predicted)
        changes, _ = analyze_decision_changes(truth, baseline, predicted)
        mismatch = np.flatnonzero(predicted != alternate)
        row = {"epsilon": float(epsilon), **metrics, **changes,
               **{f"delta_{key}": metrics[key] - baseline_metrics[key] for key in METRIC_COLUMNS},
               "mean_prudence_score": float(rho.mean()),
               "mean_real_threshold": float(real_threshold.mean()),
               "min_real_threshold": float(real_threshold.min()), "max_real_threshold": float(real_threshold.max()),
               "mean_fake_threshold": float(fake_threshold.mean()),
               "min_fake_threshold": float(fake_threshold.min()), "max_fake_threshold": float(fake_threshold.max()),
               "formulation_disagreements": len(mismatch),
               "formulation_disagreement_percent": 100.0 * len(mismatch) / len(validation),
               "passes_accuracy_constraint": metrics["accuracy"] >= baseline_metrics["accuracy"] - MAX_ACCURACY_LOSS,
               "passes_recall_constraint": metrics["fake_recall"] >= baseline_metrics["fake_recall"]}
        row["is_feasible"] = row["passes_accuracy_constraint"] and row["passes_recall_constraint"]
        rows.append(row)
        groups.extend(_sensitivity_rows(validation, predicted, float(epsilon)))
        for index in mismatch[:DISAGREEMENT_EXAMPLES_PER_EPSILON]:
            example = {"epsilon": float(epsilon), "validation_row": int(index),
                       **{c: validation.iloc[index][c] for c in ("image_id", "image_path", *PROBABILITY_COLUMNS) if c in validation},
                       "prudence_score": float(rho[index]), "real_threshold": float(real_threshold[index]),
                       "fake_threshold": float(fake_threshold[index]),
                       "form_a_prediction": predicted[index], "form_b_prediction": alternate[index]}
            examples.append(example)
    return pd.DataFrame(rows), baseline_metrics, pd.DataFrame(groups), pd.DataFrame(examples)


def select_best_epsilon(sweep):
    feasible = sweep.loc[sweep["is_feasible"]]
    if feasible.empty:
        raise InputError("No EPSILON meets both validation constraints; no parameter was selected. Consider a grid containing 0.")
    return feasible.sort_values(
        ["fake_recall", "macro_f1", "regressions", "epsilon"],
        ascending=[False, False, True, True], kind="stable",
    ).iloc[0]


def plot_epsilon_sweep(sweep, baseline, selected, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(fig, filename):
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def mark(ax):
        feasible = sweep["is_feasible"].to_numpy(dtype=bool)
        ax.fill_between(sweep["epsilon"], 0, 1, where=feasible,
                        transform=ax.get_xaxis_transform(), color="green", alpha=0.10, label="Both constraints satisfied")
        ax.axvline(selected["epsilon"], color="black", linestyle=":", label=f"BEST_EPSILON = {selected['epsilon']:.3f}")
        ax.set_xlabel("EPSILON")

    fig, ax = plt.subplots(figsize=(9, 5))
    for metric, label in (("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("fake_recall", "Fake Recall")):
        ax.plot(sweep["epsilon"], sweep[metric], label=label)
    ax.axhline(baseline["accuracy"] - MAX_ACCURACY_LOSS, color="red", linestyle="--", label="ADN Accuracy − 0.03")
    mark(ax); ax.set(title="Validation fusion sweep", ylabel="Score", ylim=(0, 1.02)); ax.legend(fontsize=8)
    save(fig, "fusion_epsilon_validation_metrics.jpg")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sweep["epsilon"], sweep["fake_fnr"], label="Fusion Fake FNR")
    ax.axhline(baseline["fake_fnr"], linestyle="--", color="gray", label="ADN Fake FNR")
    mark(ax); ax.set(title="Validation missed-Fake rate", ylabel="Fake FNR", ylim=(0, 1)); ax.legend(fontsize=8)
    save(fig, "fusion_epsilon_validation_fnr.jpg")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sweep["epsilon"], sweep["corrections"], label="Corrections")
    ax.plot(sweep["epsilon"], sweep["regressions"], label="Regressions")
    mark(ax); ax.set(title="Validation decision changes", ylabel="Images"); ax.legend(fontsize=8)
    save(fig, "fusion_epsilon_validation_decision_changes.jpg")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sweep["accuracy"], sweep["fake_recall"], color="gray", alpha=0.6)
    points = ax.scatter(sweep["accuracy"], sweep["fake_recall"], c=sweep["epsilon"], cmap="viridis", s=25)
    fig.colorbar(points, ax=ax, label="EPSILON")
    ax.scatter(baseline["accuracy"], baseline["fake_recall"], marker="X", s=130, color="blue", label="ADN baseline")
    ax.scatter(selected["accuracy"], selected["fake_recall"], marker="*", s=200, color="red", label="Selected fusion")
    ax.axvline(baseline["accuracy"] - MAX_ACCURACY_LOSS, color="red", linestyle="--", label="Accuracy constraint")
    ax.set(title="Validation Accuracy / Fake Recall trade-off", xlabel="Accuracy", ylabel="Fake Recall")
    ax.legend(fontsize=8); save(fig, "fusion_epsilon_validation_tradeoff.jpg")


def _write_json(path, payload):
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    path.write_text(json.dumps(clean(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _source_info(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def resolve_validation_path(explicit_path):
    if explicit_path:
        return Path(explicit_path).resolve()
    candidates = [folder / name for folder in OUTPUT_FOLDERS for name in VALIDATION_FILENAMES if (folder / name).is_file()]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise InputError("Multiple validation exports found; choose one explicitly with --validation:\n" + "\n".join(map(str, candidates)))
    raise InputError(
        "No per-image VALIDATION probability export was found. Aggregate validation metrics/history are insufficient.\n"
        "Missing: image_id/image_path, true_real_fake, p_real, p_fake, p_low, p_medium, p_high;\n"
        "also export true_sensitivity for group diagnostics.\n"
        "Export one inference-only pass over existing val_df using its ordered deterministic loaders and\n"
        "the already selected ADN/CSN best checkpoints. Do not retrain or resplit.\n"
        "See tools/README.md. Test fusion_predictions.csv is never a calibration substitute."
    )


def evaluate_frozen_test(test_path, validation, best_epsilon, weights, mappings, output_dir):
    """Phase B: a single frozen value, with no sweep, selection or optimization."""
    test = _read_predictions(test_path, "test", mappings)
    shared_ids = [c for c in ("image_path", "image_id") if c in validation and c in test]
    if not shared_ids:
        raise InputError("Cannot verify validation/test disjointness: no shared image identifier column.")
    for column in shared_ids:
        if set(validation[column].astype(str)) & set(test[column].astype(str)):
            raise InputError(f"Validation and test overlap in {column}; frozen test evaluation refused.")
    for column in ("training_run_id", "split_fingerprint", "adn_checkpoint", "csn_checkpoint"):
        if column in validation and column in test and validation[column].iloc[0] != test[column].iloc[0]:
            raise InputError(f"Validation/test model provenance disagrees in {column}.")
    rho = compute_prudence_score(test, weights)
    baseline, _ = fusion_from_p_real(test["p_real"], rho, 0.0)
    fusion, real_threshold = fusion_from_p_real(test["p_real"], rho, best_epsilon)
    alternate, fake_threshold = fusion_from_p_fake(test["p_fake"], rho, best_epsilon)
    changes, transitions = analyze_decision_changes(test["_authenticity"], baseline, fusion)
    summary = {"phase": "final_frozen_test_evaluation", "best_epsilon_from_validation": best_epsilon,
               "source": _source_info(test_path), "samples": len(test),
               "adn": compute_metrics(test["_authenticity"], baseline),
               "fusion_form_a": compute_metrics(test["_authenticity"], fusion), **changes,
               "formulation_disagreements": int((fusion != alternate).sum())}
    _write_json(output_dir / "fusion_frozen_test_metrics.json", summary)
    result = test.drop(columns=[c for c in test if c.startswith("_")]).copy()
    result["prudence_score"] = rho; result["epsilon"] = best_epsilon
    result["real_threshold"] = real_threshold; result["fake_threshold"] = fake_threshold
    result["adn_prediction"] = baseline; result["fusion_prediction"] = fusion
    result.to_csv(output_dir / "fusion_frozen_test_predictions.csv", index=False)
    transitions.to_csv(output_dir / "fusion_frozen_test_decision_changes.csv", index=False)
    groups = _sensitivity_rows(test, fusion, best_epsilon)
    if groups:
        pd.DataFrame(groups).to_csv(output_dir / "fusion_frozen_test_by_sensitivity.csv", index=False)
    print("\nPHASE B — FINAL FROZEN TEST EVALUATION (no selection)")
    print(pd.DataFrame([summary["adn"], summary["fusion_form_a"]], index=["ADN", "Frozen Fusion"]).round(6).to_string())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, help="Saved validation CSV/JSON; never a test export.")
    parser.add_argument("--test", type=Path, help="Optional test export, opened only after validation calibration finishes.")
    parser.add_argument("--project-notebook", type=Path, default=PROJECT_NOTEBOOK, help="Read-only source of fixed weights and label semantics.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = {"status": "input_check", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
             "validation_source": None, "test_used_for_calibration": False,
             "available_output_files": {str(folder): sorted(p.name for p in folder.iterdir() if p.is_file())
                                        for folder in OUTPUT_FOLDERS if folder.is_dir()}}
    try:
        weights, mappings = load_project_config(args.project_notebook)
        validation_path = resolve_validation_path(args.validation)
        print(f"VALIDATION SOURCE: {validation_path}\nFixed project weights: {weights}")
        validation = load_validation_predictions(validation_path, mappings)
        audit["validation_source"] = _source_info(validation_path)
        print("\nPHASE A — VALIDATION CALIBRATION ONLY; project Form A supplies selection metrics.")
        sweep, baseline, groups, examples = run_epsilon_sweep(validation, epsilon_grid(), weights)
        sweep.to_csv(args.output_dir / "fusion_epsilon_sweep_validation.csv", index=False)
        if not groups.empty:
            groups.to_csv(args.output_dir / "fusion_epsilon_by_sensitivity_validation.csv", index=False)
        if not examples.empty:
            examples.to_csv(args.output_dir / "fusion_formulation_disagreements_validation.csv", index=False)
        best = select_best_epsilon(sweep)
        BEST_EPSILON = float(best["epsilon"])
        max_error = float(np.max(np.abs(validation["p_real"] + validation["p_fake"] - 1.0)))
        complement_ok = bool(max_error <= PROBABILITY_SUM_ATOL)
        disagreement_count = int(sweep["formulation_disagreements"].sum())
        if disagreement_count == 0 and complement_ok:
            equivalence = ("For binary softmax with p_fake = 1 − p_real the forms are mathematically equivalent, "
                           "including ties. They are empirically equivalent on all saved validation samples "
                           "for every tested EPSILON.")
        elif disagreement_count == 0:
            equivalence = ("The forms agree empirically on this grid, but saved ADN probabilities fail the "
                           "complement-sum tolerance. Check the export; exact mathematical equivalence requires p_fake = 1 − p_real.")
        else:
            equivalence = ("The forms are mathematically equivalent for exact complementary probabilities, "
                           "but are NOT empirically equivalent on these saved values. Rounded/noncomplementary "
                           "values near the boundary can disagree. Selection retains the project's Form A without renormalization.")
        summary = {"phase": "validation_calibration", "validation_source": _source_info(validation_path),
                   "project_notebook": _source_info(args.project_notebook), "samples": len(validation),
                   "weights": weights, "label_semantics": mappings,
                   "grid": {"min": EPSILON_MIN, "max": EPSILON_MAX, "step": EPSILON_STEP},
                   "accuracy_loss_allowance": MAX_ACCURACY_LOSS,
                   "selection_order": ["maximum Fake Recall", "maximum Macro F1", "minimum regressions", "minimum EPSILON"],
                   "selection_formulation": "A", "best_epsilon": BEST_EPSILON,
                   "adn_baseline": baseline, "selected_fusion": best.to_dict(),
                   "max_binary_probability_sum_error": max_error,
                   "probability_sum_atol": PROBABILITY_SUM_ATOL, "binary_probability_sum_within_tolerance": complement_ok,
                   "formulation_disagreement_sample_epsilon_pairs": disagreement_count,
                   "formulation_disagreement_percent_over_all_sample_epsilon_pairs": 100 * disagreement_count / (len(validation) * len(sweep)),
                   "formulation_equivalence": equivalence, "test_used_for_calibration": False}
        _write_json(args.output_dir / "fusion_calibration_validation_summary.json", summary)
        plot_epsilon_sweep(sweep, baseline, best, args.output_dir)
        print("\nADN VALIDATION BASELINE")
        print(pd.Series(baseline).round(6).to_string())
        print(f"\nSELECTED FUSION\nBEST_EPSILON: {BEST_EPSILON:.3f}")
        print(best[[*METRIC_COLUMNS, "corrections", "regressions"]].to_string())
        print("\nDELTAS")
        print(best[[f"delta_{m}" for m in METRIC_COLUMNS]].to_string())
        print(f"\nFORMULATION CHECK\nmax |p_real + p_fake − 1|: {max_error:.10g}")
        print(sweep[["epsilon", "formulation_disagreements", "formulation_disagreement_percent"]].to_string(index=False))
        print(f"Total disagreements across sample/EPSILON pairs: {disagreement_count}\n{equivalence}")
        if not examples.empty:
            print("\nFirst disagreement examples:\n" + examples.head().to_string(index=False))
        if not groups.empty:
            print("\nSelected validation fusion by TRUE sensitivity (diagnostic only):")
            print(groups.loc[groups["epsilon"] == BEST_EPSILON].round(6).to_string(index=False))
        audit["status"] = "validation_calibration_complete"
        audit["best_epsilon"] = BEST_EPSILON
        _write_json(args.output_dir / "fusion_calibration_input_audit.json", audit)
        # Test is deliberately not loaded, inspected, or scored anywhere above.
        if args.test is not None:
            evaluate_frozen_test(args.test, validation, BEST_EPSILON, weights, mappings, args.output_dir)
        else:
            print("\nPHASE B skipped: no test predictions requested.")
        print(f"Results: {args.output_dir.resolve()}")
        return 0
    except (InputError, OSError, json.JSONDecodeError) as exc:
        audit["status"] = "blocked_missing_or_invalid_input" if audit["status"] != "validation_calibration_complete" else "frozen_test_evaluation_failed"
        audit["error"] = str(exc)
        _write_json(args.output_dir / "fusion_calibration_input_audit.json", audit)
        print(f"STOPPED: {exc}", file=sys.stderr)
        print("No training, checkpoint changes, resplitting, or test-based parameter selection occurred.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
