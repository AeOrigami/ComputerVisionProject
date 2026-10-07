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

