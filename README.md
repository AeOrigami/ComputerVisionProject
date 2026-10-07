# SensiFake

**Computer Vision Project — A.Y. 2025/2026**  
Sapienza University of Rome

SensiFake is a sensitivity-aware deepfake detection project built around two complementary objectives:

1. constructing a curated image dataset containing both **authenticity labels** (`Real` / `Fake`) and **content-sensitivity labels** (`Low` / `Medium` / `High`);
2. investigating whether sensitivity information can be used to make a deepfake detector more cautious when classification errors may be more consequential.

The project therefore covers the complete pipeline from dataset construction and sensitivity annotation to deep-learning models, external benchmarking, and a final sensitivity-aware decision policy.

---

## Dataset Construction

The SensiFake dataset combines images from three existing deepfake datasets:

| Source dataset | Selected images |
|---|---:|
| OpenFake | 1,500 |
| SID-Set | 1,500 |
| RRDataset | 1,500 source files |

The initial selection contained **4,500 source files**.

A SHA-256 audit detected one pair of byte-identical images in RRDataset, resulting in:

**4,499 unique images**

The final authenticity distribution is almost perfectly balanced:

| Authenticity | Images |
|---|---:|
| Real | 2,249 |
| Fake | 2,250 |

Authenticity labels are inherited from the original source datasets and are kept independent from the sensitivity-annotation process.

---

## Sensitivity Labels

Each image is associated with one of three content-sensitivity levels:

- **Low**
- **Medium**
- **High**

The final distribution is:

| Sensitivity | Images |
|---|---:|
| Low | 1,267 |
| Medium | 3,008 |
| High | 224 |

The strong imbalance, especially for the High class, is an important characteristic of the dataset and later affects the sensitivity-classification task.

---

## Sensitivity Annotation Pipeline

Sensitivity labels were produced through a combination of **human annotation** and **automatic annotation**.

The final dataset contains:

| Annotation source | Images |
|---|---:|
| Human annotated | 1,101 |
| Automatic-only | 3,398 |
| **Total** | **4,499** |

Human annotations were collected and reviewed using a dedicated annotation workflow.

The repository contains a Streamlit-based annotation tool supporting:

- image batch import;
- sensitivity annotation;
- annotation review;
- asynchronous work by multiple annotators;
- package management;
- annotation export;
- integration of reviewed labels into the dataset.

The annotation application and supporting modules are located in:

```text
SensiFake-Dataset_Creation/script_manual_annotator/
```

The sensitivity-model and dataset-processing code is located in:

```text
SensiFake-Dataset_Creation/script/
```

---

## Dataset Processing

The dataset-construction workflow can be summarized as:

```text
Source datasets
      │
      ▼
Image selection
      │
      ▼
Authenticity-label verification
      │
      ▼
SHA-256 audit and deduplication
      │
      ▼
Human sensitivity annotation
      │
      ▼
Sensitivity-model development
      │
      ▼
Automatic annotation of remaining images
      │
      ▼
Human review / consolidation
      │
      ▼
Unified SensiFake metadata
```

The resulting metadata preserve, when available:

- image identifier;
- image path;
- source dataset;
- Real/Fake label;
- Low/Medium/High sensitivity label;
- annotation provenance;
- SHA-256 information.

---

## Repository Structure — Dataset Components

```text
SensiFake-Dataset_Creation/
│
├── script/
│   ├── sensitivity_ordinal_v3.ipynb
│   ├── sensitivity_v2_booster.py
│   └── build_human_review_retraining_v2_manifest.py
│
├── script_manual_annotator/
│   ├── app.py
│   ├── annotation_database.py
│   ├── annotation_packages.py
│   ├── annotation_schema.py
│   ├── build_model_review_batches.py
│   ├── import_batches.py
│   └── ...
│
└── plots/
    └── sensitivity-model and dataset analyses
```

The published dataset, download instructions, provenance information, and external benchmark are described in the following section.


# DeepfakeBench Effort benchmark

`deepfakebench_xception.py` evaluates SensiFake images with DeepfakeBench's
`effort` detector. The filename is historical: this runner does **not** use
the Xception detector. It uses CLIP ViT-L/14 with Effort's trained
classification head.

The benchmark uses the manifest's verified real/fake label as the target and
reports results for:

- the complete evaluated population;
- the `low`, `medium`, and `high` sensitivity levels;
- every `source_dataset` represented in the input manifest;
- every dataset/sensitivity combination that contains images.

For published SensiFake metadata, use
`benchmark/data/sensifake-hf/metadata/sensifake_all.csv`. It contains the
resolved source dataset, source image path, verified authenticity label, and
final sensitivity label.

The evaluator imports the official DeepfakeBench Effort detector. Effort uses
CLIP ViT-L/14 at 224x224 with CLIP normalization. DeepfakeBench is kept outside
this repository.

## Download the Hugging Face dataset

Install or update the Hugging Face CLI if necessary:

```bash
uv tool install huggingface_hub
```

Download the complete published SensiFake dataset into the benchmark data
directory:

```bash
hf download K4ru4k4i/SensiFake \
  --repo-type dataset \
  --local-dir benchmark/data/sensifake-hf
```

The downloaded directory should contain
`benchmark/data/sensifake-hf/metadata/sensifake_all.csv` and the corresponding
image files referenced by that metadata. If the dataset is gated or requires
authentication, log in first:

```bash
hf auth login
```

## Setup

For the repository's local annotation manifest, first build the unified
manifest:

```bash
uv run python scripts/datasets/build_unified_manifest.py
```

Install the benchmark dependencies (run from `ComputerVisionProject`, where
`requirements.txt` lives) and obtain DeepfakeBench:

```bash
uv venv
uv pip install -r requirements.txt
git clone https://github.com/SCLBD/DeepfakeBench.git external/DeepfakeBench
mkdir -p external/DeepfakeBench/preprocessing/dlib_tools
curl -L https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.0/shape_predictor_81_face_landmarks.dat \
  -o external/DeepfakeBench/preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat
```

Download the local CLIP model used by Effort:

```bash
uv run hf download openai/clip-vit-large-patch14 \
  --local-dir external/DeepfakeBench/huggingface/clip-vit-large-patch14
```

The Effort checkpoint (about 1.2 GB) is too large for Git, so it is not
committed (`*.pth` is in `.gitignore`) and is hosted as a Hugging Face dataset
at https://huggingface.co/datasets/redaster3/effort_checkpoint. Download it
into the weights directory:

```bash
uvx --from huggingface_hub hf download redaster3/effort_checkpoint \
  effort_clip_L14_trainOn_sdv14.pth \
  --repo-type dataset \
  --local-dir external/DeepfakeBench/training/weights
```

To publish your own copy (once, after `hf auth login`):

```bash
uvx --from huggingface_hub hf upload redaster3/effort_checkpoint \
  path/to/effort_clip_L14_trainOn_sdv14.pth effort_clip_L14_trainOn_sdv14.pth \
  --repo-type dataset
```

DeepfakeBench release `v1.0.1` does not include a pretrained Effort detector
checkpoint, so use a compatible Effort checkpoint or train Effort first. Do not use
`xception_best.pth`: it is a different architecture. The checkpoint must
match the CLIP ViT-L/14 Effort model.

## Running the benchmark

The benchmark includes defaults for the published Hugging Face dataset,
DeepfakeBench checkout, local CLIP model, Effort checkpoint, and output
directory. After setup, run it from the project root
(`ComputerVisionProject`, where `.venv` lives) in Bash or WSL with:

```bash
uv run python Effort_Benchmark/benchmark/deepfakebench_xception.py
```

Relative paths passed to the script are resolved against `Effort_Benchmark`,
so the same command works from any directory if you give the full script path.

The default command uses (paths relative to `Effort_Benchmark`):

- `benchmark/data/sensifake-hf/metadata/sensifake_all.csv`;
- `external/DeepfakeBench`;
- `external/DeepfakeBench/huggingface/clip-vit-large-patch14`;
- `external/DeepfakeBench/training/weights/effort_clip_L14_trainOn_sdv14.pth`;
- `benchmark/results/deepfakebench-effort-hf`.

All paths remain configurable if a different manifest, model, checkpoint, or
output directory is needed. For example, run with the repository's local
annotation manifest:

```bash
 uv run python Effort_Benchmark/benchmark/deepfakebench_xception.py --annotations benchmark/data/sensifake-hf/metadata/sensifake_all.csv --output benchmark/results/deepfakebench-effort-severity
```
## Sensitivity-Aware Deepfake Detection

The final stage of the project investigates whether content sensitivity can be incorporated into the decision process of a deepfake detector.

The complete experiment is implemented in:

```text
ProjectSensiFake.ipynb
```

The method combines two independent neural networks:

- **ADN — Authenticity Detection Network**
- **CSN — Content Sensitivity Network**

Their probability outputs are combined through a final **Sensitivity-Aware Fusion** policy.

---

## Method Overview

The system follows the architecture:

```text
                         ┌───────────────────┐
                         │       ADN         │
Image ──────────────────►│ Authenticity      │────► P(Real), P(Fake)
                         │ Detection Network │
                         └───────────────────┘
                                   │
                                   │
                                   ▼
                           Sensitivity-Aware
                                Fusion
                                   ▲
                                   │
                         ┌───────────────────┐
                         │       CSN         │
Image ──────────────────►│ Content           │────► P(Low), P(Medium), P(High)
                         │ Sensitivity Net   │
                         └───────────────────┘
```

ADN and CSN are trained independently and receive the same input image.

The fusion is applied only after both models have produced their probability distributions.

---

## Authenticity Detection Network — ADN

ADN performs binary image classification:

```text
Real / Fake
```

It uses a pretrained **ResNet-50** backbone adapted to two output classes.

The model produces:

\[
P(Real),\qquad P(Fake)
\]

A conventional ADN prediction uses:

\[
P(Fake)\geq0.5
\]

as its Fake decision rule.

The training procedure includes:

- pretrained ResNet-50 initialization;
- progressive fine-tuning;
- dropout regularization;
- label smoothing;
- conservative data augmentation;
- validation-based checkpoint selection.

The best checkpoint is selected according to **validation Macro F1**.

---

## Content Sensitivity Network — CSN

CSN independently predicts the sensitivity level:

```text
Low / Medium / High
```

and produces:

\[
P(Low),\qquad P(Medium),\qquad P(High)
\]

CSN is also based on a pretrained **ResNet-50**.

Because High-sensitivity examples are substantially less represented, the training procedure uses class weighting and supervision weighting.

Importantly, the final fusion does not use only the CSN argmax prediction.

Instead, it uses the complete probability distribution:

\[
P(Low),P(Medium),P(High)
\]

so that uncertainty between neighboring sensitivity levels is preserved.

---

## Experimental Protocol

A single global split is shared by ADN and CSN:

| Split | Fraction |
|---|---:|
| Training | 70% |
| Validation | 10% |
| Test | 20% |

The random seed is:

```text
20
```

The test set is excluded from:

- model training;
- checkpoint selection;
- fusion calibration.

The best ADN and CSN checkpoints are selected using validation performance before final test evaluation.

The models are implemented in **PyTorch**.

---

## Sensitivity-Aware Fusion

The aim of the fusion is not to replace ADN, but to modify its decision boundary according to the predicted sensitivity of each individual image.

A continuous sensitivity or **prudence score** is defined as:

\[
\rho(x)=
\frac{1}{3}P(Low)
+
\frac{2}{3}P(Medium)
+
P(High)
\]

The fixed weights

\[
W_L=\frac13,\qquad
W_M=\frac23,\qquad
W_H=1
\]

encode the ordinal relation:

\[
Low < Medium < High
\]

The standard ADN Fake threshold of `0.5` is then modified as:

\[
\tau_{fake}(x)=0.5-\epsilon\rho(x)
\]

The final decision is:

\[
\hat y(x)=
\begin{cases}
Fake & P(Fake)\geq\tau_{fake}(x)\\
Real & P(Fake)<\tau_{fake}(x)
\end{cases}
\]

Therefore, higher predicted sensitivity lowers the threshold required to treat an image as Fake.

The result is a more conservative decision policy for sensitive content.

---

## EPSILON Calibration

The sensitivity weights are fixed by design.

Only the global fusion-strength parameter:

\[
\epsilon
\]

is calibrated from data.

Calibration is performed **exclusively on validation predictions**.

No test samples or test metrics are used to select EPSILON.

Instead of searching an arbitrary numerical grid, the final calibration computes the exact validation decision breakpoints:

\[
\epsilon_i=
\frac{0.5-P(Fake)_i}{\rho_i}
\]

These are the values at which an individual validation image changes from Real to Fake.

The 450 validation images produced:

- **202 positive decision breakpoints**
- **203 distinct decision policies**

The final selected value is:

\[
\boxed{
\epsilon=0.035769800509300682
}
\]

or approximately:

```text
EPSILON = 0.03577
```

The selected decision policy remains unchanged throughout:

\[
\boxed{
0.0357698
\leq
\epsilon
<
0.0390868
}
\]

Therefore, the validation set supports a stable optimal decision interval rather than a uniquely meaningful decimal value.

The smallest EPSILON producing the selected policy is used.

---

## Sensitivity-Aware Calibration Objective

Because sensitivity is the central element of the fusion, EPSILON is selected using **Sensitivity-Weighted Fake Recall**.

Higher-sensitivity Fake images receive greater importance during validation evaluation.

The selected policy must also satisfy:

\[
Accuracy_{Fusion}
\geq
Accuracy_{ADN}-0.03
\]

and:

\[
FakeRecall_{Fusion}
\geq
FakeRecall_{ADN}
\]

Thus, the calibration searches for a more cautious detector while limiting the allowed degradation in standard classification accuracy.

The final calibration implementation is available in:

```text
tools/calibrate_final_epsilon.py
```

Additional fusion-analysis scripts are preserved in `tools/` for reproducibility.

---

## ADN Test Results

On the held-out test set, ADN obtains:

| Metric | ADN |
|---|---:|
| Accuracy | **81.56%** |
| Macro F1 | **81.46%** |
| Fake Precision | **77.52%** |
| Fake Recall | **88.89%** |
| Fake F1 | **82.82%** |
| Fake False Negative Rate | **11.11%** |
| ROC AUC | **0.905** |

ADN therefore provides a strong Real/Fake baseline before sensitivity information is introduced.

---

## CSN Test Results

Sensitivity classification is more challenging, particularly for the minority High class.

The CSN test confusion matrix gives approximately:

| True sensitivity | Recall |
|---|---:|
| Low | **73.5%** |
| Medium | **89.9%** |
| High | **46.7%** |

The High class remains the main limitation of CSN.

This also motivates the use of the complete CSN probability distribution in the fusion instead of relying only on the predicted class.

---

## ADN vs Sensitivity-Aware Fusion

The main experimental comparison is between:

1. standard ADN with a fixed threshold of `0.5`;
2. ADN combined with the sensitivity-aware adaptive threshold.

Final test results are:

| Metric | ADN | Sensitivity-Aware Fusion |
|---|---:|---:|
| Accuracy | **81.56%** | **78.56%** |
| Fake Recall | **88.89%** | **92.22%** |
| Fake False Negative Rate | **11.11%** | **7.78%** |

Sensitivity-aware fusion therefore increases Fake Recall by approximately:

\[
+3.33\text{ percentage points}
\]

and reduces the Fake False Negative Rate by approximately:

\[
-3.33\text{ percentage points}
\]

while decreasing overall Accuracy by approximately:

\[
-3.00\text{ percentage points}
\]

---

## Analysis by Sensitivity

The effect of the fusion is not uniform across sensitivity levels.

Approximate Fake Recall on the test set is:

| True sensitivity | ADN | Fusion |
|---|---:|---:|
| Low | 81.6% | **88.8%** |
| Medium | 91.1% | **93.1%** |
| High | 97.1% | **97.1%** |

The largest change occurs for Low-sensitivity samples.

This does not mean that higher sensitivity receives less weight.

The sensitivity score determines **how much the decision threshold is shifted**, but a prediction changes only if its ADN Fake probability is sufficiently close to the original decision boundary.

ADN already detects almost all High-sensitivity Fake images, leaving little room for further improvement in that subgroup.

---

## Interpretation

The experiment does **not** show that sensitivity increases the overall classification accuracy of a deepfake detector.

Instead, it shows that sensitivity can be used to modify the detector's **operating policy**.

The resulting detector is more conservative:

- more Fake images are detected;
- fewer Fake images are incorrectly accepted as Real;
- more Real images may consequently be flagged as Fake.

The central result of the experiment can therefore be summarized as:

> **Sensitivity does not make the deepfake detector more accurate; it makes the detector more cautious.**

This trade-off can be desirable in applications where missing potentially sensitive synthetic content is considered more costly than producing additional false alarms.

---

## Main Notebook Structure

`ProjectSensiFake.ipynb` follows the project code organization:

```text
Imports
Globals
Utils
Data
Network
Train
Evaluation
```

The notebook contains:

1. SensiFake dataset loading;
2. shared train/validation/test split;
3. ADN architecture and training;
4. CSN architecture and training;
5. best-checkpoint validation;
6. ADN test evaluation;
7. CSN test evaluation;
8. sensitivity-aware fusion;
9. ADN vs Fusion comparison;
10. evaluation by true sensitivity;
11. single-image inference demo.

---

## Reproducibility

Project dependencies are listed in:

```text
requirements.txt
```

Install them with:

```bash
pip install -r requirements.txt
```

The final model checkpoints are stored in:

```text
checkpoints/
├── adn_best.pt
└── csn_best.pt
```

Fusion-calibration utilities are stored in:

```text
tools/
```

The final validation-only calibration can be run with:

```bash
python tools/calibrate_final_epsilon.py
```

---

## Limitations

The main limitations observed in the current experiments are:

- strong imbalance among sensitivity classes;
- limited representation of High-sensitivity content;
- lower CSN performance on the High class;
- a relatively small validation set for fusion calibration;
- the sensitivity-aware policy improves Fake Recall but introduces additional false positives;
- the current study is limited to still images.

---

## Future Work

Possible extensions include:

- collecting more human-annotated High-sensitivity samples;
- improving CSN performance on minority sensitivity classes;
- calibrating ADN and CSN probability estimates;
- comparing the adaptive threshold against a globally calibrated ADN threshold;
- evaluating alternative risk-sensitive objectives;
- extending the approach to video deepfake detection;
- developing an interactive application that warns the user when an image is classified as both Fake and highly sensitive.

---

## Conclusion

SensiFake investigates deepfake detection not only as a classification problem, but also as a **risk-aware decision problem**.

ADN provides the underlying authenticity estimate, while CSN adds contextual information about image sensitivity.

The final sensitivity-aware fusion does not improve overall test accuracy. Instead, it changes the operating point of the detector, increasing Fake Recall from **88.89% to 92.22%** and reducing the Fake False Negative Rate from **11.11% to 7.78%**, with a controlled reduction in Accuracy.

The experiments therefore support the use of sensitivity as a mechanism for building a **more cautious deepfake detector**, rather than as a mechanism for maximizing conventional classification accuracy.

