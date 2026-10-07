#!/usr/bin/env python3
"""Exact, validation-only EPSILON calibration with fixed ordinal sensitivity weights.

No training, inference, checkpoints, notebooks, previous calibrators, or test data.
Probabilities are read as round-trip binary64 values. Fraction arithmetic treats
those saved values and the weights 1/3, 2/3, 1 exactly. Breakpoints and policy
comparisons never use rounded EPSILON decimals or a tolerance. Decimal columns
are display approximations; accompanying numerator/denominator values are exact.
The float64 tolerance below is ONLY an audit of cancellation near a boundary.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


W_LOW = Fraction(1, 3)
W_MEDIUM = Fraction(2, 3)
W_HIGH = Fraction(1, 1)
SEVERITY_UNITS = {"low": 1, "medium": 2, "high": 3}
AUTHENTICITY_LABELS = {0: "real", 1: "fake"}
SENSITIVITY_LABELS = {0: "low", 1: "medium", 2: "high"}
EXPECTED_VALIDATION_SAMPLES = 450
MAX_ACCURACY_LOSS = 0.03
PROBABILITY_SUM_ATOL = 1e-6
FLOAT64_BOUNDARY_AUDIT_ATOL = 8 * np.finfo(np.float64).eps
RANDOM_SEED = 20
N_FOLDS = 5
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_PATH = PROJECT_ROOT / "SensiFakeProject-output-v3-validation/validation_predictions.csv"
OUTPUT_DIR = PROJECT_ROOT / "tools/epsilon_final"
REQUIRED_COLUMNS = ["image_id", "image_path", "true_real_fake", "true_sensitivity",
                    "p_real", "p_fake", "p_low", "p_medium", "p_high"]
METRIC_COLUMNS = ["accuracy", "macro_f1", "fake_precision", "fake_recall", "fake_f1", "fake_fnr",
                  "weighted_fake_recall", "weighted_fake_fnr"]
SELECTION_PRIORITY = [
    "maximum Sensitivity-Weighted Fake Recall", "maximum ordinary Fake Recall",
    "maximum Macro F1", "minimum regressions", "maximum Fake F1",
    "minimum changed predictions", "minimum exact EPSILON",
]
INTERVAL_STATEMENT = (
    "The validation data support an optimal decision interval rather than a uniquely identifiable EPSILON decimal."
)
OUTPUT_FILENAMES = [
    "final_epsilon_breakpoints_validation.csv", "final_epsilon_changed_samples.csv",
    "final_epsilon_by_true_sensitivity.csv", "final_epsilon_summary.json",
    "final_epsilon_5fold_stability.csv", "final_epsilon_stability_summary.json",
    "final_epsilon_weighted_recall.jpg", "final_epsilon_accuracy_tradeoff.jpg",
    "final_epsilon_metrics.jpg", "final_epsilon_fnr.jpg",
    "final_epsilon_decision_changes.jpg", "final_epsilon_by_sensitivity.jpg",
    "final_epsilon_console_summary.txt",
]


class CalibrationError(ValueError):
    """Invalid input or an exact-arithmetic/methodological consistency failure."""


def _fraction(value):
    return value if isinstance(value, Fraction) else Fraction.from_float(float(value))


def _exact_fields(value, prefix):
    return {f"{prefix}_numerator": str(value.numerator) if value is not None else None,
            f"{prefix}_denominator": str(value.denominator) if value is not None else None}


def _display(value):
    return "unbounded" if value is None else f"{float(value):.17g}"


def locate_validation_predictions(path=VALIDATION_PATH):
    path = Path(path).resolve()
    if path.name == "fusion_predictions.csv" or "test" in path.stem.lower():
        raise CalibrationError(f"Test artifact is forbidden as calibration input: {path}")
    if not path.is_file():
        raise CalibrationError(f"Missing validation probabilities: {path}; no test substitution is allowed.")
    return path


def _map_semantics(values, mapping, column):
    def semantic(value):
        text = str(value).strip().lower()
        if text in mapping.values():
            return text
        try:
            number = float(text)
        except (ValueError, TypeError):
            number = np.nan
        if np.isfinite(number) and number.is_integer() and int(number) in mapping:
            return mapping[int(number)]
        raise CalibrationError(f"Invalid {column} label {value!r}; retain the project label semantics.")
    return values.map(semantic)


def validate_input(frame, expected_count=EXPECTED_VALIDATION_SAMPLES):
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise CalibrationError(f"Missing validation columns: {', '.join(missing)}")
    if len(frame) != expected_count or frame.empty:
        raise CalibrationError(f"Expected {expected_count} validation samples; found {len(frame)}.")
    for column in ("image_id", "image_path"):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise CalibrationError(f"Missing {column} values.")
        if frame[column].duplicated().any():
            raise CalibrationError(f"Duplicate {column} values; one row per image is required.")
    if "split" in frame and not frame["split"].astype(str).str.lower().str.strip().isin(["val", "validation"]).all():
        raise CalibrationError("Rows outside validation are present; test data are forbidden.")
    result = frame.copy().reset_index(drop=True)
    try:
        result[REQUIRED_COLUMNS[4:]] = result[REQUIRED_COLUMNS[4:]].apply(pd.to_numeric, errors="raise").astype(float)
    except (ValueError, TypeError) as exc:
        raise CalibrationError("Probabilities must be numeric.") from exc
    probabilities = result[REQUIRED_COLUMNS[4:]].to_numpy()
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise CalibrationError("Probabilities must be finite, without NaNs, and in [0, 1].")
    checks = {}
    for task, columns in (("adn", ["p_real", "p_fake"]), ("csn", ["p_low", "p_medium", "p_high"])):
        error = np.abs(result[columns].sum(axis=1).to_numpy() - 1)
        checks[f"maximum_{task}_probability_sum_error"] = float(error.max())
        if not (error <= PROBABILITY_SUM_ATOL).all():
            raise CalibrationError(f"{task.upper()} probabilities do not sum to one within {PROBABILITY_SUM_ATOL}.")
    auth = _map_semantics(result["true_real_fake"], AUTHENTICITY_LABELS, "true_real_fake")
    sensitivity = _map_semantics(result["true_sensitivity"], SENSITIVITY_LABELS, "true_sensitivity")
    if set(auth) != {"real", "fake"}:
        raise CalibrationError("Both Real and Fake samples are required for this calibration and stratified diagnostic.")
    result["_true_fake"] = auth.eq("fake").to_numpy()
    result["_sensitivity"] = sensitivity
    result["_severity_units"] = sensitivity.map(SEVERITY_UNITS).to_numpy(dtype=int)
    for column in ("training_run_id", "split_fingerprint", "adn_checkpoint", "csn_checkpoint"):
        if column in result and (result[column].isna().any() or result[column].nunique() != 1):
            raise CalibrationError(f"Mixed or missing provenance in {column}.")
    return result, checks


def compute_prudence_score(frame):
    """Exact ordinal combination of saved binary64 CSN values."""
    scores = np.array([
        W_LOW * _fraction(low) + W_MEDIUM * _fraction(medium) + W_HIGH * _fraction(high)
        for low, medium, high in frame[["p_low", "p_medium", "p_high"]].to_numpy()
    ], dtype=object)
    if any(score <= 0 for score in scores):
        raise CalibrationError("Every valid sample must have positive prudence.")
    return scores


def compute_weighted_fake_recall(truth, predicted, severity_units):
    truth, predicted = np.asarray(truth, dtype=bool), np.asarray(predicted, dtype=bool)
    units = np.asarray(severity_units, dtype=int)
    denominator = int(units[truth].sum())
    numerator = int(units[truth & predicted].sum())
    # Dividing all severities by 3 cancels; integer counts make objective ties exact.
    return numerator / denominator if denominator else np.nan


def _exact_f1(tp, tn, fp, fn):
    fake_denominator = 2 * tp + fp + fn
    real_denominator = 2 * tn + fp + fn
    fake = Fraction(2 * tp, fake_denominator) if fake_denominator else Fraction(0)
    real = Fraction(2 * tn, real_denominator) if real_denominator else Fraction(0)
    return (fake + real) / 2, fake


def compute_metrics(truth, predicted, severity_units):
    truth, predicted = np.asarray(truth, dtype=bool), np.asarray(predicted, dtype=bool)
    tp, tn, fp, fn = (int((truth & predicted).sum()), int((~truth & ~predicted).sum()),
                      int((~truth & predicted).sum()), int((truth & ~predicted).sum()))
    macro, fake_f1 = _exact_f1(tp, tn, fp, fn)
    weighted = compute_weighted_fake_recall(truth, predicted, severity_units)
    return {"accuracy": (tp + tn) / len(truth) if len(truth) else np.nan,
            "macro_f1": float(macro), "fake_precision": tp / (tp + fp) if tp + fp else 0.0,
            "fake_recall": tp / (tp + fn) if tp + fn else np.nan,
            "fake_f1": float(fake_f1), "fake_fnr": fn / (tp + fn) if tp + fn else np.nan,
            "weighted_fake_recall": weighted, "weighted_fake_fnr": 1 - weighted}


def compute_adn_baseline(frame):
    predicted = frame["p_fake"].to_numpy() >= 0.5
    return predicted, compute_metrics(frame["_true_fake"], predicted, frame["_severity_units"])


def compute_exact_breakpoints(frame):
    rho = compute_prudence_score(frame)
    baseline, _ = compute_adn_baseline(frame)
    per_sample = np.array([None if fake else (Fraction(1, 2) - _fraction(p_fake)) / score
                           for fake, p_fake, score in zip(baseline, frame["p_fake"], rho, strict=True)], dtype=object)
    positive = [value for value in per_sample if value is not None]
    if any(value <= 0 for value in positive):
        raise CalibrationError("ADN Real samples must have strictly positive breakpoints.")
    candidates = sorted({Fraction(0), *positive})
    return candidates, per_sample, rho


def apply_fusion(frame, epsilon, per_sample_breakpoints=None, rho=None):
    """Mathematically equivalent breakpoint comparison, with exact inclusive ties.

    For an initially Real image, EPSILON >= (0.5-p_fake)/rho is exactly equivalent
    to p_fake >= 0.5-EPSILON*rho. No approximate equality expands the interval.
    """
    epsilon = _fraction(epsilon)
    if epsilon < 0:
        raise CalibrationError("EPSILON must be nonnegative.")
    if per_sample_breakpoints is None or rho is None:
        _, per_sample_breakpoints, rho = compute_exact_breakpoints(frame)
    baseline, _ = compute_adn_baseline(frame)
    predicted = baseline | np.array([point is not None and epsilon >= point for point in per_sample_breakpoints])
    thresholds = np.array([Fraction(1, 2) - epsilon * score for score in rho], dtype=object)
    # Audit direct float64 arithmetic without allowing its cancellation to change a policy.
    float_thresholds = 0.5 - float(epsilon) * np.array([float(score) for score in rho])
    literal = frame["p_fake"].to_numpy() >= float_thresholds
    disagreements = literal != predicted
    if disagreements.any():
        errors = np.abs(frame["p_fake"].to_numpy()[disagreements] - float_thresholds[disagreements])
        if not (errors <= FLOAT64_BOUNDARY_AUDIT_ATOL).all():
            raise CalibrationError("Exact and literal floating decisions disagree away from the documented boundary tolerance.")
    return predicted, thresholds, int(disagreements.sum())


def compute_decision_changes(truth, baseline, fusion, sensitivity=None):
    truth, baseline, fusion = [np.asarray(values, dtype=bool) for values in (truth, baseline, fusion)]
    corrections = (baseline != truth) & (fusion == truth)
    regressions = (baseline == truth) & (fusion != truth)
    result = {"changed_predictions": int((baseline != fusion).sum()), "corrections": int(corrections.sum()),
              "regressions": int(regressions.sum()), "real_to_fake": int((~baseline & fusion).sum()),
              "fake_to_real": int((baseline & ~fusion).sum())}
    assert result["corrections"] + result["regressions"] == result["changed_predictions"]
    assert result["real_to_fake"] + result["fake_to_real"] == result["changed_predictions"]
    if sensitivity is not None:
        for level in ("low", "medium", "high"):
            mask = np.asarray(sensitivity) == level
            result[f"corrections_{level}"] = int((corrections & mask).sum())
            result[f"regressions_{level}"] = int((regressions & mask).sum())
    return result


def collapse_identical_policies(candidates, predictions):
    policies = []
    for epsilon, predicted in zip(candidates, predictions, strict=True):
        if policies and np.array_equal(policies[-1]["predictions"], predicted):
            policies[-1]["candidate_count"] += 1
            continue
        if policies:
            policies[-1]["upper"] = epsilon
        policies.append({"lower": epsilon, "upper": None, "predictions": predicted.copy(), "candidate_count": 1})
    for policy in policies:
        assert policy["upper"] is None or policy["upper"] > policy["lower"]
    return policies


def analyze_breakpoint_policies(frame):
    candidates, per_sample, rho = compute_exact_breakpoints(frame)
    evaluated = [apply_fusion(frame, epsilon, per_sample, rho) for epsilon in candidates]
    policies = collapse_identical_policies(candidates, [item[0] for item in evaluated])
    baseline, baseline_metrics = compute_adn_baseline(frame)
    truth = frame["_true_fake"].to_numpy(dtype=bool)
    units = frame["_severity_units"].to_numpy(dtype=int)
    sensitivity = frame["_sensitivity"].to_numpy()
    baseline_correct = int((baseline == truth).sum())
    rows = []
    for policy_id, policy in enumerate(policies):
        predicted = policy["predictions"]
        metrics = compute_metrics(truth, predicted, units)
        changes = compute_decision_changes(truth, baseline, predicted, sensitivity)
        tp, tn, fp, fn = (int((truth & predicted).sum()), int((~truth & ~predicted).sum()),
                          int((~truth & predicted).sum()), int((truth & ~predicted).sum()))
        macro_exact, fake_f1_exact = _exact_f1(tp, tn, fp, fn)
        weighted_detected = int(units[truth & predicted].sum())
        exact_accuracy_constraint = Fraction(tp + tn, len(frame)) >= Fraction(baseline_correct, len(frame)) - Fraction(str(MAX_ACCURACY_LOSS))
        weighted_baseline_detected = int(units[truth & baseline].sum())
        ordinary_baseline_detected = int((truth & baseline).sum())
        policy["sort_key"] = (-weighted_detected, -tp, -macro_exact, changes["regressions"],
                              -fake_f1_exact, changes["changed_predictions"], policy["lower"])
        policy["feasible"] = bool(exact_accuracy_constraint and tp >= ordinary_baseline_detected
                                  and weighted_detected >= weighted_baseline_detected)
        _, thresholds, literal_disagreements = apply_fusion(frame, policy["lower"], per_sample, rho)
        row = {"policy_id": policy_id, "epsilon": float(policy["lower"]),
               "interval_lower": float(policy["lower"]), "interval_upper": float(policy["upper"]) if policy["upper"] is not None else np.inf,
               "interval_lower_inclusive": True, "interval_upper_inclusive": False,
               "interval_upper_unbounded": policy["upper"] is None, "merged_candidate_count": policy["candidate_count"],
               **_exact_fields(policy["lower"], "epsilon"), **_exact_fields(policy["upper"], "interval_upper"),
               **metrics, **{f"delta_{key}": metrics[key] - baseline_metrics[key] for key in METRIC_COLUMNS}, **changes,
               "weighted_true_fake_detected_units": weighted_detected,
               "weighted_true_fake_total_units": int(units[truth].sum()),
               "passes_accuracy_constraint": bool(exact_accuracy_constraint),
               "passes_fake_recall_constraint": tp >= ordinary_baseline_detected,
               "passes_weighted_recall_constraint": weighted_detected >= weighted_baseline_detected,
               "is_feasible": policy["feasible"], "is_selected": False,
               "literal_float64_boundary_disagreements": literal_disagreements,
               "minimum_fake_threshold": float(min(thresholds)), "maximum_fake_threshold": float(max(thresholds)),
               "thresholds_below_zero": sum(value < 0 for value in thresholds)}
        rows.append(row)
    return pd.DataFrame(rows), policies, baseline_metrics, candidates, per_sample, rho


def select_best_policy(table, policies):
    feasible = [index for index, policy in enumerate(policies) if policy["feasible"]]
    if not feasible:
        raise CalibrationError("No feasible policy; the EPSILON=0 baseline should satisfy all constraints.")
    index = min(feasible, key=lambda value: policies[value]["sort_key"])
    table.loc[:, "is_selected"] = table["policy_id"] == index
    return index, policies[index]


def compute_optimal_interval(selected):
    return {"BEST_EPSILON": float(selected["lower"]),
            "BEST_EPSILON_INTERVAL_LOWER": float(selected["lower"]),
            "BEST_EPSILON_INTERVAL_UPPER": float(selected["upper"]) if selected["upper"] is not None else None,
            "interval_lower_inclusive": True, "interval_upper_inclusive": False,
            "interval_upper_unbounded": selected["upper"] is None,
            **_exact_fields(selected["lower"], "BEST_EPSILON"),
            **_exact_fields(selected["upper"], "BEST_EPSILON_INTERVAL_UPPER")}


def analyze_true_sensitivity(frame, selected_predictions):
    baseline, _ = compute_adn_baseline(frame)
    rows, comparisons = [], []
    for level in ("low", "medium", "high"):
        mask = frame["_sensitivity"].eq(level).to_numpy()
        truth = frame.loc[mask, "_true_fake"].to_numpy(dtype=bool)
        units = frame.loc[mask, "_severity_units"].to_numpy()
        metrics = compute_metrics(truth, selected_predictions[mask], units) if mask.any() else {}
        baseline_metrics = compute_metrics(truth, baseline[mask], units) if mask.any() else {}
        changes = compute_decision_changes(truth, baseline[mask], selected_predictions[mask])
        rows.append({"sensitivity": level.title(), "samples": int(mask.sum()), "true_fake_samples": int(truth.sum()),
                     **{key: metrics.get(key, np.nan) for key in ("accuracy", "fake_recall", "fake_fnr")},
                     **{key: changes[key] for key in ("changed_predictions", "corrections", "regressions")}})
        for name, values in (("ADN", baseline_metrics), ("Fusion", metrics)):
            comparisons.append({"sensitivity": level.title(), "model": name,
                                "fake_recall": values.get("fake_recall", np.nan), "fake_fnr": values.get("fake_fnr", np.nan)})
    assert sum(row["samples"] for row in rows) == len(frame)
    return pd.DataFrame(rows), pd.DataFrame(comparisons)


def changed_sample_audit(frame, selected, per_sample, rho):
    baseline, _ = compute_adn_baseline(frame)
    changed = np.flatnonzero(baseline != selected["predictions"])
    order = sorted(changed, key=lambda index: (per_sample[index], str(frame.iloc[index]["image_id"])))
    audit = frame.iloc[order][["image_id", "image_path", "true_real_fake", "true_sensitivity",
                               "p_fake", "p_low", "p_medium", "p_high"]].copy()
    audit["prudence_score"] = [float(rho[index]) for index in order]
    audit["breakpoint_epsilon"] = [float(per_sample[index]) for index in order]
    audit["breakpoint_epsilon_numerator"] = [str(per_sample[index].numerator) for index in order]
    audit["breakpoint_epsilon_denominator"] = [str(per_sample[index].denominator) for index in order]
    audit["adn_prediction"] = baseline[order].astype(int)
    audit["fusion_prediction"] = selected["predictions"][order].astype(int)
    truth = frame["_true_fake"].to_numpy(dtype=bool)
    audit["correction"] = (~baseline[order] & truth[order])
    audit["regression"] = (~truth[order] & selected["predictions"][order])
    audit["changes_exactly_at_best_epsilon"] = [per_sample[index] == selected["lower"] for index in order]
    assert len(audit) == int((baseline != selected["predictions"]).sum())
    assert (audit["correction"] ^ audit["regression"]).all()
    return audit


def run_stability_analysis(frame):
    truth = frame["_true_fake"].to_numpy(dtype=bool)
    if np.bincount(truth.astype(int), minlength=2).min() < N_FOLDS:
        raise CalibrationError("Too few samples per authenticity class for the requested stratified 5-fold diagnostic.")
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    rows, heldout_rows = [], []
    for fold, (calibration_indices, heldout_indices) in enumerate(splitter.split(np.zeros(len(frame)), truth), start=1):
        assert not np.intersect1d(calibration_indices, heldout_indices).size
        calibration, heldout = frame.iloc[calibration_indices].reset_index(drop=True), frame.iloc[heldout_indices].reset_index(drop=True)
        table, policies, _, _, _, _ = analyze_breakpoint_policies(calibration)
        selected_id, selected = select_best_policy(table, policies)
        selected_row = table.loc[table["policy_id"] == selected_id].iloc[0]
        # Held-out probabilities are evaluated only after fold calibration has finished.
        predicted, _, disagreements = apply_fusion(heldout, selected["lower"])
        baseline, baseline_metrics = compute_adn_baseline(heldout)
        metrics = compute_metrics(heldout["_true_fake"], predicted, heldout["_severity_units"])
        changes = compute_decision_changes(heldout["_true_fake"], baseline, predicted)
        rows.append({"fold": fold, "calibration_samples": len(calibration), "heldout_samples": len(heldout),
                     "selected_epsilon": float(selected["lower"]),
                     **compute_optimal_interval(selected),
                     **{f"calibration_{key}": float(selected_row[key]) for key in METRIC_COLUMNS},
                     **{f"heldout_{key}": value for key, value in metrics.items()},
                     **{f"heldout_adn_{key}": value for key, value in baseline_metrics.items()}, **changes,
                     "heldout_literal_float64_boundary_disagreements": disagreements})
        heldout_rows.extend(int(value) for value in heldout_indices)
    assert sorted(heldout_rows) == list(range(len(frame)))
    result = pd.DataFrame(rows)
    summaries, baseline_summaries = {}, {}
    for key in METRIC_COLUMNS:
        values, baseline_values = result[f"heldout_{key}"], result[f"heldout_adn_{key}"]
        summaries[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
        baseline_summaries[key] = {"mean": float(baseline_values.mean()), "std": float(baseline_values.std(ddof=1))}
    summary = {"diagnostic_only": True, "folds": N_FOLDS, "random_seed": RANDOM_SEED,
               "strategy": "StratifiedKFold by true Real/Fake", "metric_std_ddof": 1,
               "selected_epsilon_values": result["selected_epsilon"].tolist(),
               "selected_exact_epsilon_values": [{"numerator": row["BEST_EPSILON_numerator"],
                                                  "denominator": row["BEST_EPSILON_denominator"]} for row in rows],
               "selected_epsilon_mean": float(result["selected_epsilon"].mean()),
               "selected_epsilon_std": float(result["selected_epsilon"].std(ddof=1)),
               "selected_epsilon_min": float(result["selected_epsilon"].min()),
               "selected_epsilon_max": float(result["selected_epsilon"].max()),
               "heldout_metrics": summaries, "heldout_adn_metrics": baseline_summaries,
               "full_validation_selection_unchanged_by_diagnostic": True, "test_used_for_calibration": False,
               "interpretation": "Resampling diagnoses sensitivity to validation membership. Calibration constraints do not guarantee held-out fold constraints."}
    return result, summary


def generate_plots(table, baseline, selected, comparisons):
    os.environ["MPLCONFIGDIR"] = str(OUTPUT_DIR / ".matplotlib_cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = table["epsilon"].to_numpy()
    finite_upper = float(selected["upper"]) if selected["upper"] is not None else x[-1] + max(0.01, x[-1] * 0.05)
    plot_end = x[-1] + max(0.01, x[-1] * 0.03)
    plot_x = np.append(x, plot_end)

    def steps(ax, column, label):
        values = table[column].to_numpy()
        ax.step(plot_x, np.append(values, values[-1]), where="post", label=label)

    def mark(ax):
        ax.axvspan(float(selected["lower"]), finite_upper, color="green", alpha=0.18, label="Selected policy interval")
        ax.axvline(float(selected["lower"]), color="black", linestyle=":", label="BEST_EPSILON")
        ax.set_xlabel("Exact EPSILON breakpoint (decimal display)")

    def save(fig, filename):
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    steps(ax, "weighted_fake_recall", "Sensitivity-Weighted Fake Recall")
    steps(ax, "fake_recall", "Ordinary Fake Recall")
    ax.scatter([0, 0], [baseline["weighted_fake_recall"], baseline["fake_recall"]], color="red", marker="X", s=70, label="ADN baseline")
    mark(ax); ax.set(title="Validation recall at exact decision policies", ylabel="Recall"); ax.legend(fontsize=8)
    save(fig, "final_epsilon_weighted_recall.jpg")

    fig, ax = plt.subplots(figsize=(8, 5))
    feasible = table["is_feasible"].to_numpy(dtype=bool)
    ax.plot(table["accuracy"], table["weighted_fake_recall"], color="gray", alpha=.5)
    ax.scatter(table["accuracy"], table["weighted_fake_recall"], s=15, color="gray", label="All policies")
    ax.scatter(table.loc[feasible, "accuracy"], table.loc[feasible, "weighted_fake_recall"], s=35, color="green", label="Feasible policies")
    row = table.loc[table["is_selected"]].iloc[0]
    ax.scatter(baseline["accuracy"], baseline["weighted_fake_recall"], marker="X", s=140, color="blue", label="ADN baseline")
    ax.scatter(row["accuracy"], row["weighted_fake_recall"], marker="*", s=180, color="red", label="Selected policy")
    ax.axvline(baseline["accuracy"] - MAX_ACCURACY_LOSS, linestyle="--", color="red", label="Accuracy lower bound")
    ax.set(title="Validation Accuracy / weighted recall trade-off", xlabel="Accuracy", ylabel="Sensitivity-Weighted Fake Recall")
    ax.legend(fontsize=8); save(fig, "final_epsilon_accuracy_tradeoff.jpg")

    fig, ax = plt.subplots(figsize=(10, 5))
    for column, label in (("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("fake_recall", "Fake Recall"),
                          ("weighted_fake_recall", "Sensitivity-Weighted Fake Recall")):
        steps(ax, column, label)
    ax.axhline(baseline["accuracy"] - MAX_ACCURACY_LOSS, color="red", linestyle="--", label="Accuracy constraint")
    mark(ax); ax.set(title="Validation metrics: exact breakpoint policies", ylabel="Score"); ax.legend(fontsize=8)
    save(fig, "final_epsilon_metrics.jpg")

    fig, ax = plt.subplots(figsize=(9, 4))
    steps(ax, "fake_fnr", "Ordinary Fake FNR"); steps(ax, "weighted_fake_fnr", "Sensitivity-Weighted Fake FNR")
    mark(ax); ax.set(title="Missed-Fake importance across decision policies", ylabel="FNR"); ax.legend(fontsize=8)
    save(fig, "final_epsilon_fnr.jpg")

    fig, ax = plt.subplots(figsize=(9, 4))
    for column, label in (("corrections", "Corrections"), ("regressions", "Regressions"), ("changed_predictions", "Changed predictions")):
        steps(ax, column, label)
    mark(ax); ax.set(title="Validation decision changes", ylabel="Images"); ax.legend(fontsize=8)
    save(fig, "final_epsilon_decision_changes.jpg")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in zip(axes, ("fake_recall", "fake_fnr"), ("Fake Recall", "Fake FNR")):
        data = comparisons.pivot(index="sensitivity", columns="model", values=metric)
        data.reindex(["Low", "Medium", "High"])[["ADN", "Fusion"]].plot.bar(ax=ax, rot=0)
        ax.set(title=title, xlabel="True sensitivity", ylabel="Rate", ylim=(0, 1))
        ax.legend(title="Model")
    save(fig, "final_epsilon_by_sensitivity.jpg")


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
    parser.add_argument("--validation", type=Path, default=VALIDATION_PATH, help="Validation CSV only; test artifacts are rejected.")
    args = parser.parse_args(argv)
    try:
        occupied = [str(OUTPUT_DIR / name) for name in OUTPUT_FILENAMES if (OUTPUT_DIR / name).exists()]
        if occupied:
            raise CalibrationError("Refusing to overwrite existing output files:\n" + "\n".join(occupied))
        path = locate_validation_predictions(args.validation)
        frame, checks = validate_input(pd.read_csv(path, float_precision="round_trip"))
        table, policies, baseline, candidates, per_sample, rho = analyze_breakpoint_policies(frame)
        selected_id, selected = select_best_policy(table, policies)
        BEST_EPSILON = selected["lower"]  # Exact Fraction, never a rounded grid value.
        selected_row = table.loc[table["policy_id"] == selected_id].iloc[0]
        sensitivity, comparisons = analyze_true_sensitivity(frame, selected["predictions"])
        audit = changed_sample_audit(frame, selected, per_sample, rho)
        stability, stability_summary = run_stability_analysis(frame)
        interval = compute_optimal_interval(selected)
        metrics = {key: float(selected_row[key]) for key in METRIC_COLUMNS}
        deltas = {key: metrics[key] - baseline[key] for key in METRIC_COLUMNS}
        at_boundary = audit.loc[audit["changes_exactly_at_best_epsilon"]]
        interval_message = INTERVAL_STATEMENT if selected["upper"] is None or selected["upper"] > BEST_EPSILON else "The selected policy is isolated."
        summary = {
            "validation_input_path": str(path), "validation_input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "sample_count": len(frame), "input_checks": checks,
            "fixed_sensitivity_weights": {"low": "1/3", "medium": "2/3", "high": "1"},
            "severity_source": "TRUE sensitivity labels of TRUE Fake samples; predicted CSN probabilities are used only in prudence",
            "label_semantics": {"authenticity": AUTHENTICITY_LABELS, "sensitivity": SENSITIVITY_LABELS},
            "adn_baseline_metrics": baseline,
            "adn_sensitivity_weighted_metrics": {key: baseline[key] for key in ("weighted_fake_recall", "weighted_fake_fnr")},
            "number_of_exact_breakpoints": len(candidates) - 1, "number_of_epsilon_candidates_including_zero": len(candidates),
            "number_of_unique_decision_policies": len(policies), "number_of_feasible_policies": int(table["is_feasible"].sum()),
            "selection_constraints": {"maximum_absolute_accuracy_loss": MAX_ACCURACY_LOSS,
                                      "accuracy_lower_bound": baseline["accuracy"] - MAX_ACCURACY_LOSS,
                                      "minimum_fake_recall": baseline["fake_recall"],
                                      "minimum_weighted_fake_recall": baseline["weighted_fake_recall"]},
            "selection_priority": SELECTION_PRIORITY, **interval, "optimal_policy_interpretation": interval_message,
            "selected_fusion_metrics": metrics, "all_deltas_vs_adn": deltas,
            "changed_predictions": int(selected_row["changed_predictions"]),
            "corrections": int(selected_row["corrections"]), "regressions": int(selected_row["regressions"]),
            "corrections_regressions_by_true_sensitivity": {level: {key: int(selected_row[f"{key}_{level}"]) for key in ("corrections", "regressions")}
                                                           for level in ("low", "medium", "high")},
            "samples_changing_exactly_at_best_epsilon": at_boundary[["image_id", "image_path", "true_real_fake", "true_sensitivity"]].to_dict("records"),
            "true_sensitivity_diagnostics": sensitivity.to_dict("records"),
            "numerical_policy": {"probability_reader": "binary64 round-trip parsing",
                                 "breakpoint_and_decision_arithmetic": "exact rational arithmetic; numerator/denominator fields are authoritative",
                                 "decimal_epsilon_fields": "nearest binary64 display approximation, not used to select or evaluate policies",
                                 "epsilon_comparison_tolerance": 0, "probability_sum_atol": PROBABILITY_SUM_ATOL,
                                 "literal_float64_boundary_audit_atol": FLOAT64_BOUNDARY_AUDIT_ATOL,
                                 "literal_float64_boundary_disagreements_at_selected_policy": int(selected_row["literal_float64_boundary_disagreements"]),
                                 "literal_float64_disagreement_sample_policy_pairs": int(table["literal_float64_boundary_disagreements"].sum())},
            "full_validation_selected_before_stability_analysis": True,
            "stability_used_for_final_selection": False, "parameter_selected_using_validation_only": True,
            "test_used_for_calibration": False, "stability": stability_summary,
        }
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(OUTPUT_DIR / "final_epsilon_breakpoints_validation.csv", index=False, float_format="%.17g")
        audit.to_csv(OUTPUT_DIR / "final_epsilon_changed_samples.csv", index=False, float_format="%.17g")
        sensitivity.to_csv(OUTPUT_DIR / "final_epsilon_by_true_sensitivity.csv", index=False, float_format="%.17g")
        stability.to_csv(OUTPUT_DIR / "final_epsilon_5fold_stability.csv", index=False, float_format="%.17g")
        save_summary("final_epsilon_summary.json", summary)
        save_summary("final_epsilon_stability_summary.json", stability_summary)
        generate_plots(table, baseline, selected, comparisons)
        messages = ["FIXED SENSITIVITY WEIGHTS\nLow: 1/3\nMedium: 2/3\nHigh: 1",
                    "\nADN VALIDATION BASELINE", pd.Series(baseline).to_string(),
                    "\nFINAL SENSITIVITY-AWARE FUSION", f"BEST_EPSILON: {_display(BEST_EPSILON)}",
                    f"Exact BEST_EPSILON: {BEST_EPSILON.numerator}/{BEST_EPSILON.denominator}",
                    f"Optimal interval: [{_display(selected['lower'])}, {_display(selected['upper'])})",
                    pd.Series(metrics).to_string(),
                    f"Changed predictions: {summary['changed_predictions']}\nCorrections: {summary['corrections']}\nRegressions: {summary['regressions']}",
                    interval_message, "\nDELTAS VS ADN", pd.Series(deltas).to_string(),
                    "\nSTABILITY", "5-fold selected EPSILON values: " + ", ".join(f"{value:.17g}" for value in stability_summary["selected_epsilon_values"])]
        for key in ("accuracy", "macro_f1", "fake_recall", "weighted_fake_recall", "fake_fnr", "weighted_fake_fnr"):
            values = stability_summary["heldout_metrics"][key]
            messages.append(f"5-fold held-out {key}: {values['mean']:.8f} ± {values['std']:.8f}")
        messages.extend([f"Parameter variability: min={stability_summary['selected_epsilon_min']:.17g}, "
                         f"max={stability_summary['selected_epsilon_max']:.17g}, sample std={stability_summary['selected_epsilon_std']:.8g}",
                         "\nTRUE-SENSITIVITY DIAGNOSTICS", sensitivity.to_string(index=False),
                         "\nSAMPLES CHANGING EXACTLY AT BEST_EPSILON", at_boundary[["image_id", "image_path"]].to_string(index=False),
                         f"Exact breakpoints: {len(candidates)-1}; candidate values including zero: {len(candidates)}; unique policies: {len(policies)}",
                         f"Literal float64 boundary disagreements at selected policy: {summary['numerical_policy']['literal_float64_boundary_disagreements_at_selected_policy']}",
                         "Breakpoints and decisions use exact rational comparisons, without EPSILON rounding or tolerance-based policy expansion.",
                         "TEST USED FOR CALIBRATION: NO", f"Outputs: {OUTPUT_DIR}"])
        console = "\n".join(messages) + "\n"
        print(console)
        (OUTPUT_DIR / "final_epsilon_console_summary.txt").write_text(console, encoding="utf-8")
        return 0
    except (CalibrationError, OSError, json.JSONDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        print(f"STOPPED: {exc}\nTEST USED FOR CALIBRATION: NO", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
