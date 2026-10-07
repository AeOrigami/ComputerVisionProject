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

