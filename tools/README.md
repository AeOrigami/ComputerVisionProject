# Offline fusion calibration

`calibrate_fusion.py` reads saved probabilities only. It does not import PyTorch,
load models/checkpoints/images, train, run inference, or execute notebook cells.
It reads fixed sensitivity weights and label semantics from
`SensiFakeProject-v3.ipynb` using Python's syntax tree. Defaults are an EPSILON
grid of 0 to 0.5 in increments of 0.005, with the required absolute accuracy-loss
allowance of 0.03. Edit the grid constants at the top of the script if needed.

## Current input availability

No per-image validation probabilities were found in:

- `SensiFakeProject-output-v2/`
- `SensiFakeProject-output-v3-validation/`
- `SensiFakeProject-output/`

The first two directories have aggregate validation metrics and per-epoch
training histories. These do not contain image identifiers, per-image true
authenticity/sensitivity labels, or `p_real`, `p_fake`, `p_low`, `p_medium`, and
`p_high`. Aggregate metrics cannot reconstruct those probabilities.

`SensiFakeProject-output-v2/fusion_predictions.csv` and
`SensiFakeProject-output/fusion_predictions.csv` are **test** exports. The script
explicitly refuses them as validation inputs. No BEST_EPSILON can be reported
from the currently available artifacts.

## Minimal prerequisite: export existing validation inference

This export is separate from the offline analysis and has not been performed.
Do not change or rerun training to obtain it.

1. Choose one existing model pair. For the complete updated notebook's run,
   use `SensiFakeProject-output-v2/adn_best.pt` and `csn_best.pt`. The checkpoints
   in `SensiFakeProject-output-v3-validation/` belong to a separate validation
   run; do not combine models or outputs from the two runs.
2. Reuse the already established ordered `val_df`, class mappings, loaded best
   models, and ADN/CSN validation loaders from that run. The `_evaluate` helper
   currently retains labels and class predictions, but not softmax probabilities.
3. Run one inference-only pass over each validation loader, with both models in
   evaluation mode, the existing deterministic transform, and
   `torch.inference_mode()`. Keep the notebook's existing AMP behavior. Save the
   softmax probabilities as float32 or float64 values with enough CSV precision;
   never reconstruct probabilities from predicted classes or aggregate metrics.
4. Assert both loaders' image paths equal `val_df.image_path` in the same order,
   both lengths equal `len(val_df)`, and collected labels match the existing
   mappings. Write exactly one row per validation image to
   `SensiFakeProject-output-v2/fusion_validation_predictions.csv` with:

   ```text
   image_id,image_path,true_real_fake,true_sensitivity,p_real,p_fake,p_low,p_medium,p_high
   ```

   Either image identifier column is sufficient; true sensitivity is optional
   for calibration and required for the optional sensitivity diagnostics.
   Add `split=validation` and, where available, `training_run_id`,
   `split_fingerprint`, `adn_checkpoint`, and `csn_checkpoint` to document provenance.
   Preserve the project's label semantics: 0=Real, 1=Fake; 0=Low, 1=Medium, 2=High.
   The corresponding named labels are also accepted.

If the live notebook session is unavailable, an export needs the existing exact
validation membership and the read-only best checkpoints. Do not substitute a
new split or initialize a new training run to satisfy notebook provenance guards.
The saved configuration records the deterministic seed and split algorithm,
but the offline calibration script never rebuilds a split itself.

## Run

Use a Python environment with the project's existing NumPy, pandas,
scikit-learn, and Matplotlib dependencies:

```bash
python tools/calibrate_fusion.py
python tools/calibrate_fusion.py --validation SensiFakeProject-output-v2/fusion_validation_predictions.csv
```

With no validation export, the script exits with status 2 and writes only an
input audit to `formula_test_output/fusion_calibration_input_audit.json`.
When several validation exports exist, supply `--validation` to select one.
JSON inputs may be row-record arrays or an object containing a `predictions`
row-record array. Aggregate validation JSON metrics are not suitable inputs.

The sweep preserves the project's strict Real condition, `p_real > 0.5 + epsilon*rho`;
ties are Fake. It explicitly tests the alternative `p_fake >= 0.5 - epsilon*rho`.
The ADN baseline is Form A at EPSILON=0. Saved probabilities are never normalized,
and thresholds are never clipped. The complement check uses an absolute tolerance
of 0.001 to accommodate previously saved AMP outputs and reports the actual maximum
error. Even a small rounding error can cause a boundary disagreement, which is
reported per EPSILON. Selection always uses the project's Form A.

Only validation values determine BEST_EPSILON, using the two required constraints
and the exact recall/F1/regressions/smaller-EPSILON ordering. Sensitivity groups
are diagnostics; empty groups and groups without true Fake samples use NaN.
Validation without any true Fake samples cannot support the recall constraint.

## Outputs

All results go into `formula_test_output/`, separate from notebook artifacts:

- `fusion_epsilon_sweep_validation.csv`
- `fusion_calibration_validation_summary.json`
- `fusion_calibration_input_audit.json`
- `fusion_epsilon_validation_metrics.jpg`
- `fusion_epsilon_validation_fnr.jpg`
- `fusion_epsilon_validation_decision_changes.jpg`
- `fusion_epsilon_validation_tradeoff.jpg`
- `fusion_epsilon_by_sensitivity_validation.csv` when true sensitivity is present
- `fusion_formulation_disagreements_validation.csv` when disagreements occur

Reruns replace this script's own corresponding results; unrelated artifacts are
not overwritten. Input SHA256 values, weights, grid, constraints, and the source
notebook are recorded for reproducibility.

## Optional frozen test evaluation

Test evaluation is off by default. After validation calibration succeeds:

```bash
python tools/calibrate_fusion.py \
  --validation SensiFakeProject-output-v2/fusion_validation_predictions.csv \
  --test SensiFakeProject-output-v2/fusion_predictions.csv
```

Use predictions from the **same checkpoint pair** as the validation export.
Matching provenance fields are checked when present; legacy CSVs without those
fields require the caller to establish that they came from the same models.
The script checks validation/test identifiers are disjoint. It opens test
predictions only after the validation sweep, selection, summary, and plots have
completed. It evaluates one frozen EPSILON and writes separate
`fusion_frozen_test_*.json/csv` files. It never performs a test EPSILON sweep.
