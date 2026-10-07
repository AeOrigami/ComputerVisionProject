#!/usr/bin/env python3
"""Validation-only offline search for monotonic fusion weights and EPSILON.

No models, inference, checkpoints, original data split, or test artifacts are loaded.
Run from any directory using the project's Python environment.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


# Fixed scale removes the weight/EPSILON identifiability redundancy.
W_H = 1.0
W_L_VALUES = np.arange(0.0, 0.65, 0.05)
W_M_VALUES = np.arange(0.20, 1.01, 0.05)
EPSILON_VALUES = np.arange(0.0, 0.151, 0.005)
MAX_ACCURACY_LOSS = 0.03
PREVIOUS_CALIBRATED_EPSILON = 0.05  # Original, unnormalized weights.
EXPECTED_VALIDATION_SAMPLES = 450
PROBABILITY_SUM_ATOL = 1e-6
N_FOLDS = 5
RANDOM_SEED = 20
SEARCH_CHUNK_SIZE = 512
TOP_CONFIGURATIONS = 20
# Predeclared final tie-break, not fitted to subgroup or held-out performance.
SMOOTH_W_LOW = 1.0 / 3.0
SMOOTH_W_MEDIUM = 2.0 / 3.0
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_PATH = PROJECT_ROOT / "SensiFakeProject-output-v3-validation/validation_predictions.csv"
NOTEBOOK_PATH = PROJECT_ROOT / "SensiFakeProject-v3-validation.ipynb"
OUTPUT_DIR = PROJECT_ROOT / "tools/formula_wigth_test_output"
REQUIRED_COLUMNS = [
    "image_id", "image_path", "true_real_fake", "true_sensitivity",
    "p_real", "p_fake", "p_low", "p_medium", "p_high",
]
METRIC_COLUMNS = ["accuracy", "macro_f1", "fake_precision", "fake_recall", "fake_f1", "fake_fnr"]
PARAMETER_COLUMNS = ["w_low", "w_medium", "w_high", "epsilon"]
SORT_COLUMNS = ["fake_recall", "macro_f1", "regressions", "fake_f1", "epsilon",
                "smoothness_penalty", "total_weight", "w_low", "w_medium"]
SORT_ASCENDING = [False, False, True, False, True, True, True, True, True]
SELECTION_PRIORITY = [
    "maximum Fake Recall", "maximum Macro F1", "minimum regressions",
    "maximum Fake F1", "minimum EPSILON",
    "minimum squared distance to (W_L=1/3, W_M=2/3, W_H=1)",
    "minimum W_L + W_M", "minimum W_L", "minimum W_M",
]


class CalibrationError(ValueError):
    """Invalid validation input or inconsistent experiment assumptions."""


def read_project_conventions():
    """Read literal conventions from the notebook without executing its code."""
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    values, mappings = {}, {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                if target.id in ("W_L", "W_M", "W_H", "EPSILON", "RANDOM_SEED"):
                    values[target.id] = ast.literal_eval(node.value)
                if target.id in ("adn_indices", "csn_indices"):
                    call = node.value
                    if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                            and call.func.id == "_semantic_indices"):
                        mappings[target.id] = ast.literal_eval(call.args[1])
    if set(mappings) != {"adn_indices", "csn_indices"}:
        raise CalibrationError("Could not read the existing project label conventions.")
    if values.get("RANDOM_SEED") != RANDOM_SEED:
        raise CalibrationError("Diagnostic seed differs from the existing project's seed.")
    if any(not np.isclose(values.get(name, np.nan), expected, rtol=0, atol=1e-12)
           for name, expected in (("W_L", 0.30), ("W_M", 0.50), ("W_H", 0.93))):
        raise CalibrationError("Current heuristic weights differ from the requested 0.30/0.50/0.93 reference.")
    return values, mappings


def _semantic_labels(series, mapping, column):
    def convert(value):
        text = str(value).strip().lower()
        if text in mapping.values():
            return text
        try:
            numeric = float(text)
        except (ValueError, TypeError):
            numeric = np.nan
        if np.isfinite(numeric) and numeric.is_integer() and int(numeric) in mapping:
            return mapping[int(numeric)]
        raise CalibrationError(f"Invalid {column} label {value!r}; retain the project label semantics.")
    return series.map(convert)


def validate_input(frame, mappings):
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise CalibrationError(f"Missing required validation columns: {', '.join(missing)}")
    if len(frame) != EXPECTED_VALIDATION_SAMPLES:
        raise CalibrationError(f"Expected {EXPECTED_VALIDATION_SAMPLES} validation images, found {len(frame)}.")
    for column in ("image_id", "image_path"):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise CalibrationError(f"Missing {column} values.")
        if frame[column].duplicated().any():
            raise CalibrationError(f"Duplicate {column} values; expected one row per validation image.")
    if "split" in frame and not frame["split"].astype(str).str.lower().str.strip().isin(["val", "validation"]).all():
        raise CalibrationError("The input contains rows outside validation; test data are forbidden.")
    result = frame.copy().reset_index(drop=True)
    probability_columns = REQUIRED_COLUMNS[4:]
    try:
        result[probability_columns] = result[probability_columns].apply(pd.to_numeric, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise CalibrationError("Saved probabilities must be numeric.") from exc
    probabilities = result[probability_columns].to_numpy()
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise CalibrationError("Probabilities must be finite, without NaNs, and lie in [0, 1].")
    errors = {}
    for task, columns in (("adn", ["p_real", "p_fake"]), ("csn", ["p_low", "p_medium", "p_high"])):
        error = np.abs(result[columns].sum(axis=1).to_numpy() - 1)
        errors[f"max_{task}_probability_sum_error"] = float(error.max())
        if not (error <= PROBABILITY_SUM_ATOL).all():
            raise CalibrationError(f"{task.upper()} probabilities do not sum to one within {PROBABILITY_SUM_ATOL}.")
    authenticity = _semantic_labels(result["true_real_fake"], mappings["adn_indices"], "true_real_fake")
    sensitivity = _semantic_labels(result["true_sensitivity"], mappings["csn_indices"], "true_sensitivity")
    if set(authenticity) != {"real", "fake"}:
        raise CalibrationError("Both Real and Fake validation samples are required for calibration and 5-fold diagnostics.")
    if set(mappings["csn_indices"].values()) != {"low", "medium", "high"}:
        raise CalibrationError("Unexpected project sensitivity semantics.")
    result["_true_fake"] = authenticity.eq("fake").to_numpy()
    result["_true_sensitivity"] = sensitivity
    for column in ("training_run_id", "split_fingerprint", "adn_checkpoint", "csn_checkpoint"):
        if column in result and (result[column].isna().any() or result[column].nunique() != 1):
            raise CalibrationError(f"Mixed or missing provenance in {column}.")
    return result, errors


def load_validation_predictions(path, mappings):
    path = Path(path).resolve()
    if path.name == "fusion_predictions.csv" or "test" in path.stem.lower():
        raise CalibrationError(f"Refusing test artifact as input: {path}")
    if not path.is_file():
        raise CalibrationError(f"Validation input does not exist: {path}; no test substitution is allowed.")
    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise CalibrationError(f"Invalid validation CSV: {exc}") from exc
    return validate_input(frame, mappings)


def _metrics_from_counts(tp, tn, fp, fn):
    tp, tn, fp, fn = [np.asarray(value, dtype=float) for value in (tp, tn, fp, fn)]
    def divide(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    fake_f1 = divide(2 * tp, 2 * tp + fp + fn)
    return {
        "accuracy": divide(tp + tn, tp + tn + fp + fn),
        "macro_f1": (fake_f1 + divide(2 * tn, 2 * tn + fp + fn)) / 2,
        "fake_precision": divide(tp, tp + fp),
        "fake_recall": np.divide(tp, tp + fn, out=np.full_like(tp, np.nan), where=(tp + fn) != 0),
        "fake_f1": fake_f1,
        "fake_fnr": np.divide(fn, tp + fn, out=np.full_like(tp, np.nan), where=(tp + fn) != 0),
    }


def compute_metrics(truth, predicted):
    truth, predicted = np.asarray(truth, dtype=bool), np.asarray(predicted, dtype=bool)
    counts = ((truth & predicted).sum(), (~truth & ~predicted).sum(),
              (~truth & predicted).sum(), (truth & ~predicted).sum())
    return {key: float(value) for key, value in _metrics_from_counts(*counts).items()}


def compute_adn_baseline(frame):
    predicted = frame["p_fake"].to_numpy() >= 0.5
    return predicted, compute_metrics(frame["_true_fake"], predicted)


def compute_prudence_score(frame, w_low, w_medium, w_high=W_H):
    return (w_low * frame["p_low"].to_numpy() + w_medium * frame["p_medium"].to_numpy()
            + w_high * frame["p_high"].to_numpy())


def apply_fusion(frame, w_low, w_medium, epsilon, w_high=W_H):
    rho = compute_prudence_score(frame, w_low, w_medium, w_high)
    threshold = 0.5 - epsilon * rho
    # Out-of-range thresholds retain their mathematical meaning. Never clip.
    return frame["p_fake"].to_numpy() >= threshold, rho, threshold


def analyze_decision_changes(truth, baseline, fusion):
    truth, baseline, fusion = [np.asarray(v, dtype=bool) for v in (truth, baseline, fusion)]
    changed = baseline != fusion
    result = {
        "changed_predictions": int(changed.sum()),
        "real_to_fake": int((~baseline & fusion).sum()),
        "fake_to_real": int((baseline & ~fusion).sum()),
        "corrections": int(((baseline != truth) & (fusion == truth)).sum()),
        "regressions": int(((baseline == truth) & (fusion != truth)).sum()),
    }
    assert result["corrections"] + result["regressions"] == result["changed_predictions"]
    assert result["real_to_fake"] + result["fake_to_real"] == result["changed_predictions"]
    return result


def _grid_values(values, name):
    values = np.unique(np.round(np.asarray(values, dtype=float), 12))
    if not len(values) or not np.isfinite(values).all() or (values < 0).any():
        raise CalibrationError(f"Invalid {name} grid: values must be finite and nonnegative.")
    return values


def generate_parameter_grid():
    if W_H != 1.0:
        raise CalibrationError("W_H must remain fixed at 1.0 to remove scale redundancy.")
    lows, mediums, epsilons = (_grid_values(W_L_VALUES, "W_L"), _grid_values(W_M_VALUES, "W_M"),
                               _grid_values(EPSILON_VALUES, "EPSILON"))
    if (lows > 1).any() or (mediums > 1).any():
        raise CalibrationError("Weight grid values must lie in [0, 1].")
    records = [(low, medium, W_H, epsilon) for low in lows for medium in mediums
               if low <= medium for epsilon in epsilons]
    if not records:
        raise CalibrationError("No monotonic weight combinations exist in the grid.")
    return pd.DataFrame(records, columns=PARAMETER_COLUMNS)


def run_grid_search(frame, parameter_grid):
    """Vectorize each chunk across configurations and validation samples."""
    baseline, baseline_metrics = compute_adn_baseline(frame)
    truth = frame["_true_fake"].to_numpy(dtype=bool)
    if not truth.any():
        raise CalibrationError("Calibration subset has no true Fake samples.")
    sens = frame[["p_low", "p_medium", "p_high"]].to_numpy()
    p_fake = frame["p_fake"].to_numpy()
    blocks = []
    for start in range(0, len(parameter_grid), SEARCH_CHUNK_SIZE):
        block = parameter_grid.iloc[start:start + SEARCH_CHUNK_SIZE].copy()
        rho = block[["w_low", "w_medium", "w_high"]].to_numpy() @ sens.T
        thresholds = 0.5 - block["epsilon"].to_numpy()[:, None] * rho
        predicted = p_fake[None, :] >= thresholds
        tp = (predicted & truth).sum(axis=1)
        tn = (~predicted & ~truth).sum(axis=1)
        fp = (predicted & ~truth).sum(axis=1)
        fn = (~predicted & truth).sum(axis=1)
        metrics = _metrics_from_counts(tp, tn, fp, fn)
        for metric, value in metrics.items():
            block[metric] = value
            block[f"delta_{metric}"] = value - baseline_metrics[metric]
        changed = predicted != baseline[None, :]
        block["changed_predictions"] = changed.sum(axis=1)
        block["real_to_fake"] = (predicted & ~baseline).sum(axis=1)
        block["fake_to_real"] = (~predicted & baseline).sum(axis=1)
        block["corrections"] = ((predicted == truth) & (baseline != truth)).sum(axis=1)
        block["regressions"] = ((predicted != truth) & (baseline == truth)).sum(axis=1)
        assert (block["corrections"] + block["regressions"] == block["changed_predictions"]).all()
        assert (block["fake_to_real"] == 0).all(), "Nonnegative prudence can only lower the Fake threshold."
        for prefix, values in (("prudence_score", rho), ("fake_threshold", thresholds)):
            for label, function in (("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)):
                block[f"{label}_{prefix}"] = function(values, axis=1)
        block["thresholds_below_zero"] = (thresholds < 0).sum(axis=1)
        block["thresholds_above_one"] = (thresholds > 1).sum(axis=1)
        block["thresholds_outside_unit_interval"] = (thresholds < 0).any(axis=1) | (thresholds > 1).any(axis=1)
        block["passes_accuracy_constraint"] = block["accuracy"] >= baseline_metrics["accuracy"] - MAX_ACCURACY_LOSS
        block["passes_recall_constraint"] = block["fake_recall"] >= baseline_metrics["fake_recall"]
        block["is_feasible"] = block["passes_accuracy_constraint"] & block["passes_recall_constraint"]
        block["smoothness_penalty"] = ((block["w_low"] - SMOOTH_W_LOW) ** 2
                                       + (block["w_medium"] - SMOOTH_W_MEDIUM) ** 2)
        block["total_weight"] = block["w_low"] + block["w_medium"]
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True), baseline_metrics


def rank_feasible_configurations(sweep):
    feasible = sweep.loc[sweep["is_feasible"]]
    if feasible.empty:
        raise CalibrationError("No feasible configuration satisfies both validation constraints.")
    return feasible.sort_values(SORT_COLUMNS, ascending=SORT_ASCENDING, kind="stable")


def select_best_configuration(sweep):
    return rank_feasible_configurations(sweep).iloc[0]


def evaluate_configuration(frame, configuration):
    params = {key: float(configuration[key]) for key in PARAMETER_COLUMNS}
    fusion, rho, thresholds = apply_fusion(frame, params["w_low"], params["w_medium"], params["epsilon"], params["w_high"])
    baseline, baseline_metrics = compute_adn_baseline(frame)
    metrics = compute_metrics(frame["_true_fake"], fusion)
    return {**params, **metrics, **{f"delta_{key}": metrics[key] - baseline_metrics[key] for key in METRIC_COLUMNS},
            **analyze_decision_changes(frame["_true_fake"], baseline, fusion),
            "mean_prudence_score": float(rho.mean()), "std_prudence_score": float(rho.std(ddof=0)),
            "min_prudence_score": float(rho.min()), "max_prudence_score": float(rho.max()),
            "mean_fake_threshold": float(thresholds.mean()), "std_fake_threshold": float(thresholds.std(ddof=0)),
            "min_fake_threshold": float(thresholds.min()), "max_fake_threshold": float(thresholds.max()),
            "thresholds_below_zero": int((thresholds < 0).sum()), "thresholds_above_one": int((thresholds > 1).sum())}


def compare_current_configuration(frame, conventions):
    original = {"w_low": float(conventions["W_L"]), "w_medium": float(conventions["W_M"]),
                "w_high": float(conventions["W_H"])}
    normalized_weights = {key: value / original["w_high"] for key, value in original.items()}
    references, checks = {}, {}
    for name, original_epsilon in (("current_original", float(conventions["EPSILON"])),
                                   ("current_calibrated", PREVIOUS_CALIBRATED_EPSILON)):
        normalized = {**normalized_weights, "epsilon": original_epsilon * original["w_high"]}
        original_params = {**original, "epsilon": original_epsilon}
        before, _, before_threshold = apply_fusion(frame, original["w_low"], original["w_medium"], original_epsilon, original["w_high"])
        after, _, after_threshold = apply_fusion(frame, normalized["w_low"], normalized["w_medium"], normalized["epsilon"])
        errors = np.abs(before_threshold - after_threshold)
        if not np.allclose(before_threshold, after_threshold, rtol=0, atol=1e-14) or not np.array_equal(before, after):
            raise CalibrationError("Normalization failed to reproduce the original effective Fake threshold/predictions.")
        # Diagnostic only: do not optimize the old Real formulation again.
        original_real_threshold = 0.5 + original_epsilon * compute_prudence_score(frame, **original)
        original_real_prediction_fake = ~(frame["p_real"].to_numpy() > original_real_threshold)
        checks[name] = {"original_representation": original_params, "normalized_representation": normalized,
                        "max_threshold_normalization_error": float(errors.max()),
                        "normalization_prediction_disagreements": int((before != after).sum()),
                        "saved_p_real_vs_preferred_p_fake_disagreements": int((original_real_prediction_fake != after).sum())}
        references[name] = evaluate_configuration(frame, normalized)
    return references, checks


def analyze_true_sensitivity(frame, configuration):
    baseline, _ = compute_adn_baseline(frame)
    fusion, _, _ = apply_fusion(frame, configuration["w_low"], configuration["w_medium"], configuration["epsilon"])
    rows = []
    for level in ("low", "medium", "high"):
        mask = frame["_true_sensitivity"].eq(level).to_numpy()
        metrics = compute_metrics(frame.loc[mask, "_true_fake"], fusion[mask]) if mask.any() else {}
        changes = analyze_decision_changes(frame.loc[mask, "_true_fake"], baseline[mask], fusion[mask])
        rows.append({"sensitivity": level.title(), "samples": int(mask.sum()),
                     **{key: metrics.get(key, np.nan) for key in ("accuracy", "fake_recall", "fake_fnr")},
                     **{key: changes[key] for key in ("changed_predictions", "corrections", "regressions")}})
    assert sum(row["samples"] for row in rows) == len(frame)
    return pd.DataFrame(rows)


def run_stability_analysis(frame, parameter_grid):
    truth = frame["_true_fake"].to_numpy()
    counts = np.bincount(truth.astype(int), minlength=2)
    if counts.min() >= N_FOLDS:
        splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        splits = splitter.split(np.zeros(len(frame)), truth)
        strategy = "StratifiedKFold by true Real/Fake"
    else:
        splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        splits = splitter.split(frame)
        strategy = "KFold fallback: insufficient samples per authenticity class for stratification"
    rows, memberships = [], []
    for fold, (calibration_indices, heldout_indices) in enumerate(splits, start=1):
        assert not np.intersect1d(calibration_indices, heldout_indices).size
        assert len(calibration_indices) + len(heldout_indices) == len(frame)
        calibration = frame.iloc[calibration_indices]
        heldout = frame.iloc[heldout_indices]
        sweep, calibration_baseline = run_grid_search(calibration, parameter_grid)
        selected = select_best_configuration(sweep)
        evaluation = evaluate_configuration(heldout, selected)
        _, heldout_baseline = compute_adn_baseline(heldout)
        rows.append({"fold": fold, "calibration_samples": len(calibration), "heldout_samples": len(heldout),
                     **{key: float(selected[key]) for key in PARAMETER_COLUMNS},
                     "calibration_accuracy": float(selected["accuracy"]),
                     "calibration_macro_f1": float(selected["macro_f1"]),
                     "calibration_fake_recall": float(selected["fake_recall"]),
                     "calibration_adn_accuracy": calibration_baseline["accuracy"],
                     "calibration_adn_fake_recall": calibration_baseline["fake_recall"],
                     **{f"heldout_{key}": evaluation[key] for key in METRIC_COLUMNS},
                     **{f"heldout_adn_{key}": heldout_baseline[key] for key in METRIC_COLUMNS},
                     **{key: evaluation[key] for key in ("changed_predictions", "corrections", "regressions", "real_to_fake", "fake_to_real")},
                     "heldout_delta_accuracy": evaluation["delta_accuracy"],
                     "heldout_delta_fake_recall": evaluation["delta_fake_recall"]})
        memberships.extend({"fold": fold, "validation_row": int(index),
                            "image_id": frame.iloc[index]["image_id"], "role": role}
                           for role, indices in (("calibration", calibration_indices), ("heldout", heldout_indices))
                           for index in indices)
    result = pd.DataFrame(rows)
    heldout_memberships = [row for row in memberships if row["role"] == "heldout"]
    assert sorted(row["validation_row"] for row in heldout_memberships) == list(range(len(frame)))
    summary = {"folds": N_FOLDS, "seed": RANDOM_SEED, "strategy": strategy,
               "diagnostic_only": True, "final_parameters_selected_on_full_validation": True,
               "heldout_metric_std_ddof": 1, "heldout_metric_summary": {}, "heldout_adn_metric_summary": {},
               "selected_parameter_frequencies": {},
               "test_used_for_calibration": False,
               "interpretation": "Fold scores are robustness diagnostics on saved validation outputs, not independent final test estimates."}
    for key in ("accuracy", "macro_f1", "fake_recall", "fake_fnr"):
        values = result[f"heldout_{key}"]
        summary["heldout_metric_summary"][key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
        adn_values = result[f"heldout_adn_{key}"]
        summary["heldout_adn_metric_summary"][key] = {"mean": float(adn_values.mean()), "std": float(adn_values.std(ddof=1))}
    for key in ("w_low", "w_medium", "epsilon"):
        summary["selected_parameter_frequencies"][key] = {
            f"{float(value):.12g}": int(count) for value, count in result[key].value_counts().sort_index().items()}
    summary["mean_heldout_delta_accuracy_vs_adn"] = float(result["heldout_delta_accuracy"].mean())
    summary["mean_heldout_delta_fake_recall_vs_adn"] = float(result["heldout_delta_fake_recall"].mean())
    summary["unique_selected_configurations"] = len(result[PARAMETER_COLUMNS].drop_duplicates())
    summary["parameter_stability_interpretation"] = (
        f"{summary['unique_selected_configurations']} distinct configurations were selected across {N_FOLDS} folds. "
        "Local plateaus in full-validation metrics do not establish stable exact weight estimates. "
        "The 0.03 Accuracy allowance is enforced on calibration subsets, not on held-out diagnostic folds."
    )
    return result, summary, pd.DataFrame(memberships)


def interpret_parameter_region(sweep, selected, frame):
    def spacing(values):
        values = np.unique(np.round(values, 12))
        return float(np.diff(values).min()) if len(values) > 1 else 0.0
    mask = sweep["is_feasible"].copy()
    for key, values in (("w_low", W_L_VALUES), ("w_medium", W_M_VALUES), ("epsilon", EPSILON_VALUES)):
        mask &= (sweep[key] - selected[key]).abs() <= spacing(values) + 1e-12
    neighbors = sweep.loc[mask & ~sweep["is_selected"]]
    fake_count = int(frame["_true_fake"].sum())
    near = ((neighbors["accuracy"] - selected["accuracy"]).abs() <= 1 / len(frame) + 1e-12)
    near &= (neighbors["fake_recall"] - selected["fake_recall"]).abs() <= 1 / fake_count + 1e-12
    near &= (neighbors["macro_f1"] - selected["macro_f1"]).abs() <= 1 / len(frame) + 1e-12
    identical = np.isclose(neighbors[METRIC_COLUMNS].to_numpy(), selected[METRIC_COLUMNS].to_numpy(dtype=float),
                           rtol=0, atol=1e-12).all(axis=1)
    many_neighbors = int(near.sum()) >= 2
    message = (
        "Several neighboring configurations yield nearly identical validation metrics: the solution lies in a stable-performing "
        "local parameter region on the full validation grid, rather than supporting uniquely optimal exact decimals."
        if many_neighbors else
        "Few immediate grid neighbors have nearly identical validation metrics; this alone does not establish a robust exact optimum."
    )
    message += " Interpret weight gaps alongside the 5-fold parameter variability; small decimal differences are not scientific evidence."
    return {"semantic_order_preserved": bool(0 <= selected["w_low"] <= selected["w_medium"] <= selected["w_high"] == 1),
            "medium_minus_low": float(selected["w_medium"] - selected["w_low"]),
            "high_minus_medium": float(selected["w_high"] - selected["w_medium"]),
            "feasible_immediate_neighbors": len(neighbors), "identical_metric_neighbors": int(identical.sum()),
            "nearly_identical_metric_neighbors": int(near.sum()),
            "near_metric_tolerances": {"accuracy": 1 / len(frame), "macro_f1": 1 / len(frame), "fake_recall": 1 / fake_count},
            "message": message}


def generate_plots(sweep, baseline, current, selected, top):
    # Keep Matplotlib's generated font/config cache inside the authorized output folder too.
    os.environ["MPLCONFIGDIR"] = str(OUTPUT_DIR / ".matplotlib_cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / name, dpi=200, bbox_inches="tight")
        plt.close(fig)

    selected_epsilon = sweep.loc[sweep["epsilon"] == selected["epsilon"]]
    for metric, title, filename in (
        ("accuracy", "Validation Accuracy", "fusion_weights_accuracy_heatmap.jpg"),
        ("fake_recall", "Validation Fake Recall", "fusion_weights_fake_recall_heatmap.jpg"),
        ("macro_f1", "Validation Macro F1", "fusion_weights_macro_f1_heatmap.jpg"),
    ):
        matrix = selected_epsilon.pivot(index="w_low", columns="w_medium", values=metric).sort_index(ascending=False)
        matrix = matrix.reindex(columns=sorted(matrix.columns))
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.heatmap(matrix, mask=matrix.isna(), annot=True, fmt=".3f", cmap="viridis", ax=ax,
                    xticklabels=[f"{v:.2f}" for v in matrix.columns], yticklabels=[f"{v:.2f}" for v in matrix.index])
        x = list(matrix.columns).index(selected["w_medium"]) + 0.5
        y = list(matrix.index).index(selected["w_low"]) + 0.5
        ax.scatter(x, y, marker="*", s=180, edgecolor="black", color="red")
        ax.set(title=f"{title} at EPSILON={selected['epsilon']:.3f}; star = selected", xlabel="W_M", ylabel="W_L")
        save(fig, filename)

    feasible = sweep.loc[sweep["is_feasible"]]
    fig, ax = plt.subplots(figsize=(9, 6))
    points = ax.scatter(feasible["accuracy"], feasible["fake_recall"], c=feasible["epsilon"],
                        cmap="viridis", alpha=0.35, s=15, label="Feasible grid configurations")
    fig.colorbar(points, ax=ax, label="EPSILON")
    for metrics, marker, color, label in ((baseline, "X", "blue", "ADN baseline"),
                                        (current, "D", "orange", "Current weights + calibrated EPSILON"),
                                        (selected, "*", "red", "Selected weights + EPSILON")):
        ax.scatter(metrics["accuracy"], metrics["fake_recall"], marker=marker, color=color,
                   s=150, edgecolors="black", label=label, zorder=5)
    ax.axvline(baseline["accuracy"] - MAX_ACCURACY_LOSS, color="gray", linestyle="--", label="Accuracy lower bound")
    ax.set(title="Validation Accuracy / Fake Recall trade-off", xlabel="Accuracy", ylabel="Fake Recall")
    ax.legend(fontsize=8); save(fig, "fusion_weights_tradeoff.jpg")

    shown = top.head(10).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(shown))
    ax.bar(x - .2, shown["corrections"], width=.4, label="Corrections")
    ax.bar(x + .2, shown["regressions"], width=.4, label="Regressions")
    labels = [f"L={row.w_low:.2f}, M={row.w_medium:.2f}\ne={row.epsilon:.3f}" for row in shown.itertuples()]
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set(title="Top feasible configurations (first = selected)", ylabel="Validation images")
    ax.legend(); save(fig, "fusion_weights_corrections_regressions.jpg")

    # Profile the selected weight pair and the closest valid neighboring pairs.
    pairs = sweep[["w_low", "w_medium"]].drop_duplicates().copy()
    pairs["distance"] = (pairs["w_low"] - selected["w_low"]) ** 2 + (pairs["w_medium"] - selected["w_medium"]) ** 2
    pairs = pairs.sort_values(["distance", "w_low", "w_medium"]).head(4)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for row in pairs.itertuples():
        profile = sweep.loc[(sweep["w_low"] == row.w_low) & (sweep["w_medium"] == row.w_medium)].sort_values("epsilon")
        label = f"W_L={row.w_low:.2f}, W_M={row.w_medium:.2f}"
        axes[0].plot(profile["epsilon"], profile["accuracy"], label=label)
        axes[1].plot(profile["epsilon"], profile["fake_recall"], label=label)
    axes[0].axhline(baseline["accuracy"] - MAX_ACCURACY_LOSS, color="gray", linestyle="--", label="Accuracy constraint")
    for ax, metric in zip(axes, ("Accuracy", "Fake Recall")):
        ax.axvline(selected["epsilon"], color="black", linestyle=":", label="Selected EPSILON")
        ax.set(xlabel="EPSILON", ylabel=metric, title=f"Nearby weights: validation {metric}")
        ax.legend(fontsize=7)
    save(fig, "fusion_weights_epsilon_profile.jpg")


def save_summary(filename, payload):
    def clean(value):
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    (OUTPUT_DIR / filename).write_text(json.dumps(clean(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, default=VALIDATION_PATH,
                        help="Saved validation export; test artifacts are rejected.")
    args = parser.parse_args(argv)
    try:
        conventions, mappings = read_project_conventions()
        frame, input_checks = load_validation_predictions(args.validation, mappings)
        parameter_grid = generate_parameter_grid()
        print(f"Validation source: {args.validation.resolve()}\nValidation samples: {len(frame)}")
        print(f"Probability-sum checks: {input_checks}")
        print(f"Searching {len(parameter_grid)} configurations with W_H fixed at 1.0.", flush=True)
        sweep, baseline = run_grid_search(frame, parameter_grid)
        selected = select_best_configuration(sweep)
        sweep["is_selected"] = sweep.index == selected.name
        top = rank_feasible_configurations(sweep).head(max(20, TOP_CONFIGURATIONS)).copy()
        top.insert(0, "rank", np.arange(1, len(top) + 1))
        selected_evaluation = evaluate_configuration(frame, selected)
        # Independent scalar recomputation must agree with vectorized search.
        for key in METRIC_COLUMNS:
            assert np.isclose(selected[key], selected_evaluation[key], rtol=0, atol=1e-12)
        references, normalization_checks = compare_current_configuration(frame, conventions)
        current = references["current_calibrated"]
        sensitivity = analyze_true_sensitivity(frame, selected)
        print("Running deterministic 5-fold validation-only stability diagnostic.", flush=True)
        stability, stability_summary, memberships = run_stability_analysis(frame, parameter_grid)
        interpretation = interpret_parameter_region(sweep, selected, frame)
        BEST_W_L, BEST_W_M, BEST_W_H, BEST_EPSILON = (float(selected[key]) for key in PARAMETER_COLUMNS)
        summary = {
            "validation_input_path": str(args.validation.resolve()),
            "validation_input_sha256": hashlib.sha256(args.validation.read_bytes()).hexdigest(),
            "sample_count": len(frame), "input_checks": input_checks,
            "project_conventions_source": str(NOTEBOOK_PATH), "label_semantics": mappings,
            "adn_baseline_metrics": baseline,
            "search_ranges": {"w_low": _grid_values(W_L_VALUES, "W_L").tolist(),
                              "w_medium": _grid_values(W_M_VALUES, "W_M").tolist(),
                              "w_high_fixed": W_H, "epsilon": _grid_values(EPSILON_VALUES, "EPSILON").tolist()},
            "tested_configurations": len(sweep), "feasible_configurations": int(sweep["is_feasible"].sum()),
            "selection_constraints": {"maximum_absolute_accuracy_loss": MAX_ACCURACY_LOSS,
                                      "accuracy_lower_bound": baseline["accuracy"] - MAX_ACCURACY_LOSS,
                                      "minimum_fake_recall": baseline["fake_recall"]},
            "selection_priority": SELECTION_PRIORITY,
            "current_normalized_configuration": normalization_checks["current_calibrated"]["normalized_representation"],
            "current_configuration_metrics": current,
            "current_original_epsilon_reference": references["current_original"],
            "normalization_checks": normalization_checks,
            "BEST_W_L": BEST_W_L, "BEST_W_M": BEST_W_M, "BEST_W_H": BEST_W_H, "BEST_EPSILON": BEST_EPSILON,
            "selected_fusion_metrics": selected_evaluation,
            "metric_deltas_vs_adn": {key: selected_evaluation[f"delta_{key}"] for key in METRIC_COLUMNS},
            "corrections": selected_evaluation["corrections"], "regressions": selected_evaluation["regressions"],
            "changed_decisions": selected_evaluation["changed_predictions"],
            "thresholds_outside_unit_interval_configurations": int(sweep["thresholds_outside_unit_interval"].sum()),
            "threshold_handling": "Use the formula without clipping. Below 0 predicts all valid probabilities Fake; above 1 predicts all Real.",
            "prudence_and_threshold_std_ddof": 0,
            "interpretability": interpretation, "stability": stability_summary,
            "final_selection_uses_full_validation_only": True,
            "true_sensitivity_analysis_used_for_selection": False,
            "stability_used_for_final_selection": False, "test_used_for_calibration": False,
        }
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        sweep.to_csv(OUTPUT_DIR / "fusion_weight_epsilon_sweep_validation.csv", index=False)
        top.to_csv(OUTPUT_DIR / "top_fusion_weight_configurations.csv", index=False)
        sensitivity.to_csv(OUTPUT_DIR / "best_weights_by_true_sensitivity.csv", index=False)
        stability.to_csv(OUTPUT_DIR / "fusion_weight_5fold_stability.csv", index=False)
        memberships.to_csv(OUTPUT_DIR / "fusion_weight_5fold_membership.csv", index=False)
        comparison = pd.DataFrame([
            {"configuration": "ADN baseline", **baseline, "changed_predictions": 0, "corrections": 0, "regressions": 0},
            {"configuration": "Current heuristic weights + calibrated EPSILON", **current},
            {"configuration": "New calibrated weights + EPSILON", **selected_evaluation},
            {"configuration": "Original heuristic EPSILON reference", **references["current_original"]},
        ])
        comparison.to_csv(OUTPUT_DIR / "adn_current_vs_calibrated_weights_validation.csv", index=False)
        save_summary("fusion_weight_calibration_summary.json", summary)
        save_summary("fusion_weight_stability_summary.json", stability_summary)
        generate_plots(sweep, baseline, current, selected, top)
        messages = ["ADN VALIDATION BASELINE", pd.Series(baseline).round(6).to_string(),
                    "\nCURRENT FUSION CONFIGURATION (normalized; original EPSILON=0.05)",
                    pd.Series({key: current[key] for key in [*PARAMETER_COLUMNS, *METRIC_COLUMNS, "corrections", "regressions"]}).round(6).to_string(),
                    "\nCALIBRATED FUSION CONFIGURATION",
                    f"BEST_W_L: {BEST_W_L:.6g}\nBEST_W_M: {BEST_W_M:.6g}\nBEST_W_H: {BEST_W_H:.6g}\nBEST_EPSILON: {BEST_EPSILON:.6g}",
                    pd.Series({key: selected_evaluation[key] for key in [*METRIC_COLUMNS, "corrections", "regressions"]}).round(6).to_string(),
                    "\nDELTAS VS ADN", pd.Series(summary["metric_deltas_vs_adn"]).round(6).to_string(),
                    "\nSTABILITY (held-out validation; mean ± sample std)"]
        for metric, values in stability_summary["heldout_metric_summary"].items():
            messages.append(f"5-fold {metric}: {values['mean']:.6f} ± {values['std']:.6f}")
        messages.extend(["Selected-parameter variability: " + json.dumps(stability_summary["selected_parameter_frequencies"]),
                         f"Held-out ADN mean Accuracy: {stability_summary['heldout_adn_metric_summary']['accuracy']['mean']:.6f}; "
                         f"mean Fusion delta Accuracy: {stability_summary['mean_heldout_delta_accuracy_vs_adn']:.6f}",
                         stability_summary["parameter_stability_interpretation"],
                         "\nTRUE-SENSITIVITY DIAGNOSTIC", sensitivity.round(6).to_string(index=False),
                         f"\nSemantic order preserved: {interpretation['semantic_order_preserved']}; "
                         f"W_M − W_L = {interpretation['medium_minus_low']:.6g}; "
                         f"W_H − W_M = {interpretation['high_minus_medium']:.6g}",
                         interpretation["message"],
                         f"Normalization prediction disagreements: {[v['normalization_prediction_disagreements'] for v in normalization_checks.values()]}",
                         f"Thresholds outside [0, 1]: {summary['thresholds_outside_unit_interval_configurations']} configurations (no clipping).",
                         "TEST USED FOR CALIBRATION: NO", f"Outputs: {OUTPUT_DIR}"])
        console = "\n".join(messages) + "\n"
        print(console)
        (OUTPUT_DIR / "fusion_weight_calibration_console_summary.txt").write_text(console, encoding="utf-8")
        return 0
    except (CalibrationError, OSError, json.JSONDecodeError) as exc:
        print(f"STOPPED: {exc}\nTEST USED FOR CALIBRATION: NO", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
