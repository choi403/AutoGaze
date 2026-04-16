<div align="center">

# AutoGaze

[![Website](https://img.shields.io/badge/Website-76b900?style=for-the-badge&logo=safari&labelColor=555555)](https://autogaze.github.io/)
[![Arxiv](https://img.shields.io/badge/Arxiv-b31b1b?style=for-the-badge&logo=arxiv&labelColor=555555)](https://arxiv.org/abs/2603.12254)
[![Models & Data & Benchmark](https://img.shields.io/badge/Models%20%26%20Data%20%26%20Benchmark-ffd21e?style=for-the-badge&logo=huggingface&labelColor=555555)](https://huggingface.co/collections/bfshi/autogaze)
[![Demo](https://img.shields.io/badge/Demo-ff6e00?style=for-the-badge&logo=huggingface&labelColor=555555)](https://huggingface.co/spaces/bfshi/AutoGaze)

</div>

AutoGaze (Autoregressive Gazing) predicts which video patches are worth keeping so downstream vision encoders or MLLMs can process fewer tokens with minimal information loss. This repository contains the official AutoGaze codebase plus a reproducible reconstruction-evaluation workflow for running the released model on real videos, measuring quality/speed tradeoffs, and generating plots and qualitative visualizations.

## What This Fork Adds

- A strict `transformers` compatibility pin: `4.51.x`. The upstream loose spec can otherwise resolve to incompatible later releases.
- An exact reproduction path for the tested machine using:
  - [`repro/micromamba-linux-64.explicit.txt`](repro/micromamba-linux-64.explicit.txt) for the conda-side packages,
  - [`repro/pip-third-party-exact.txt`](repro/pip-third-party-exact.txt) for the pip-side packages,
  - an exact FlashAttention wheel URL and checksum,
  - pinned Hugging Face revisions for all model assets used by the evaluator.
- A reproducible evaluator at [`scripts/evaluate_reconstruction.py`](scripts/evaluate_reconstruction.py) that:
  - downloads the official `VideoMAE_AutoGaze` checkpoint,
  - runs AutoGaze on real clips,
  - reconstructs the clips with the released task model,
  - computes metrics such as reconstruction loss, pixel MAE, pixel MSE, PSNR, SSIM, and runtime,
  - writes CSV summaries, plots, and qualitative PNG grids.
- Example output artifacts committed under [`analysis_outputs/reconstruction_eval`](analysis_outputs/reconstruction_eval).

## Tested Environment

The setup below was verified on this class of machine:

- Repository commit: `ba278ad70c6dda2f023230d2cb70555e6c8d6141`
- OS: Linux x86_64
- GPUs: 4x NVIDIA H100 80GB
- Driver: `580.95.05`
- CUDA driver runtime reported by `nvidia-smi`: `13.0`
- `micromamba`: `2.5.0`
- CUDA toolkit used for the environment: `12.8`
- Python: `3.11.15`
- PyTorch: `2.7.1+cu128`
- TorchVision: `0.22.1+cu128`
- FlashAttention: `2.8.3`
- Transformers: `4.51.3`

If your machine differs materially, keep the same version relationships unless you are prepared to debug CUDA and `flash-attn` ABI issues yourself.

## Reproducibility Contract

If you follow the exact reset-from-zero recipe below on a Linux `x86_64` NVIDIA machine with working CUDA, a fresh agent should be able to:

- recreate the Python/CUDA environment to the same package versions,
- pull the same Hugging Face model revisions used here,
- run the same evaluator script with the same default settings,
- regenerate the same artifact types and match the checked-in metrics to the displayed precision on the tested machine class.

What this does not promise:

- bit-for-bit identity of every floating-point tensor across different GPUs, drivers, or CUDA library stacks,
- exact runtime values on different hardware,
- compatibility with CPU-only machines.

## Pinned Upstream Asset Revisions

These are the exact Hugging Face revisions used by the evaluator:

| Repo | Revision |
|---|---|
| `nvidia/AutoGaze` | `5100fae739ec1bf3f875914fa1b703846a18943a` |
| `bfshi/VideoMAE_AutoGaze` | `34001937344859e687d336cd44ec8962018ae46b` |
| `facebook/vit-mae-large` | `142cb8c25e1b1bc1769997a919aa1b5a2345a6b8` |
| `facebook/dinov2-with-registers-base` | `a1d738ccfa7ae170945f210395d99dde8adb1805` |
| `google/siglip2-base-patch16-224` | `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2` |

## Repository Layout

- [`QUICK_START.md`](QUICK_START.md): upstream usage examples for AutoGaze and SigLIP integration.
- [`INTEGRATION.md`](INTEGRATION.md): how to adapt other ViTs/MLLMs to AutoGaze.
- [`TRAIN.md`](TRAIN.md): official training pipeline and dataset structure.
- [`scripts/evaluate_reconstruction.py`](scripts/evaluate_reconstruction.py): added evaluation script for systematic reconstruction analysis.
- [`analysis_outputs/reconstruction_eval`](analysis_outputs/reconstruction_eval): example outputs produced by the evaluation script on the bundled demo video.

## 1. Clone The Repository

For exact reproduction, use the pinned repository commit rather than a floating branch tip:

```bash
git clone https://github.com/choi403/AutoGaze.git
cd AutoGaze
git checkout ba278ad70c6dda2f023230d2cb70555e6c8d6141
```

## 2. Create The Environment

### Exact Reset-From-Zero Reproduction

This is the exact path a new agent should use if the goal is to recreate the tested environment rather than merely get something working.

Reset local state:

```bash
export REPO_ROOT="$PWD"
export HF_HOME="$REPO_ROOT/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

rm -rf "$HOME/.micromamba/envs/autogaze"
rm -rf "$REPO_ROOT/analysis_outputs/reconstruction_eval"
rm -rf "$REPO_ROOT/hf_cache"
mkdir -p "$HF_HOME"
```

Bootstrap `micromamba` if needed:

```bash
mkdir -p "$HOME/.local/bin"
cd "$HOME/.local/bin"
curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba --strip-components=1
```

Create the exact conda-side environment from the checked-in explicit lock:

```bash
MAMBA_ROOT_PREFIX="$HOME/.micromamba" \
"$HOME/.local/bin/micromamba" create -y -p "$HOME/.micromamba/envs/autogaze" \
  --file "$REPO_ROOT/repro/micromamba-linux-64.explicit.txt"
```

Install the exact tested PyTorch build:

```bash
MAMBA_ROOT_PREFIX="$HOME/.micromamba" \
"$HOME/.local/bin/micromamba" run -p "$HOME/.micromamba/envs/autogaze" \
  python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.1 torchvision==0.22.1
```

Install the exact tested FlashAttention wheel and verify its checksum:

```bash
curl -L -o "$REPO_ROOT/repro/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl" \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl

echo "cd1a45ebfc1731a13e55ad68e0c9ad92390ddfffba306f9222be67c6d5a805af  $REPO_ROOT/repro/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl" | sha256sum -c -

MAMBA_ROOT_PREFIX="$HOME/.micromamba" \
"$HOME/.local/bin/micromamba" run -p "$HOME/.micromamba/envs/autogaze" \
  python -m pip install "$REPO_ROOT/repro/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
```

Install the exact third-party pip packages, then install this repo itself without re-resolving dependencies:

```bash
MAMBA_ROOT_PREFIX="$HOME/.micromamba" \
"$HOME/.local/bin/micromamba" run -p "$HOME/.micromamba/envs/autogaze" \
  python -m pip install -r "$REPO_ROOT/repro/pip-third-party-exact.txt"

MAMBA_ROOT_PREFIX="$HOME/.micromamba" \
"$HOME/.local/bin/micromamba" run -p "$HOME/.micromamba/envs/autogaze" \
  python -m pip install -e . --no-deps
```

If you intentionally want a looser setup rather than the exact one used here, you can still use your own `conda` environment and install the pinned versions manually. That is not the recommended path for reproduction.

## 3. Verify The Installation

Run:

```bash
MAMBA_ROOT_PREFIX="$HOME/.micromamba" \
"$HOME/.local/bin/micromamba" run -p "$HOME/.micromamba/envs/autogaze" \
python - <<'PY'
import torch, torchvision, transformers, flash_attn
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("transformers", transformers.__version__)
print("flash_attn", flash_attn.__version__)
print("cuda_available", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

Expected version family:

- `torch 2.7.1+cu128`
- `torchvision 0.22.1+cu128`
- `transformers 4.51.3`
- `flash_attn 2.8.3`
- `cuda_available True`

Optional package verification:

```bash
MAMBA_ROOT_PREFIX="$HOME/.micromamba" \
"$HOME/.local/bin/micromamba" run -p "$HOME/.micromamba/envs/autogaze" \
python -m pip freeze | grep -E '^(torch|torchvision|flash_attn|transformers|timm|wandb|pandas|scikit-image|seaborn)==|^flash_attn @'
```

## 4. Model Assets

### AutoGaze weights

The released AutoGaze model is hosted at:

- [`nvidia/AutoGaze`](https://huggingface.co/nvidia/AutoGaze)

The code examples below pin it to the exact tested revision.

### Reconstruction task weights

The evaluator also needs the released reconstruction model:

- [`bfshi/VideoMAE_AutoGaze`](https://huggingface.co/bfshi/VideoMAE_AutoGaze)

The evaluation script downloads this automatically into `hf_cache/VideoMAE_AutoGaze` by default, pinned to the revision listed above.

### Training data

If you want to train rather than only run inference/evaluation, follow [`TRAIN.md`](TRAIN.md) and download:

- `bfshi/AutoGaze-Training-Data`
- `bfshi/VideoMAE_AutoGaze`

## 5. Run AutoGaze On A Video

The repository ships with a real example video at [`assets/example_input.mp4`](assets/example_input.mp4). The following script runs the released model on the first 16 frames:

```bash
MAMBA_ROOT_PREFIX="$HOME/.micromamba" \
"$HOME/.local/bin/micromamba" run -p "$HOME/.micromamba/envs/autogaze" \
python - <<'PY'
import av
import torch
from autogaze.datasets.video_utils import read_video_pyav, transform_video_for_pytorch
from autogaze.models.autogaze import AutoGazeImageProcessor, AutoGaze

video_path = "assets/example_input.mp4"
revision = "5100fae739ec1bf3f875914fa1b703846a18943a"
processor = AutoGazeImageProcessor.from_pretrained("nvidia/AutoGaze", revision=revision)
model = AutoGaze.from_pretrained("nvidia/AutoGaze", revision=revision).cuda().eval()

container = av.open(video_path)
indices = list(range(model.config.max_num_frames))
raw_video = read_video_pyav(container, indices)
container.close()

video = transform_video_for_pytorch(raw_video, processor)[None].cuda()

with torch.inference_mode():
    outputs = model({"video": video}, gazing_ratio=0.75, task_loss_requirement=0.7)

print("video_shape:", tuple(video.shape))
print("gazing_pos_shape:", tuple(outputs["gazing_pos"].shape))
print("if_padded_gazing_shape:", tuple(outputs["if_padded_gazing"].shape))
print("num_non_padded_gazes:", int((~outputs["if_padded_gazing"]).sum().item()))
print("num_gazing_each_frame:", outputs["num_gazing_each_frame"].tolist())
print("scales:", outputs["scales"])
print("num_vision_tokens_each_frame:", outputs["num_vision_tokens_each_frame"])
PY
```

On the bundled demo clip, the tested output was:

```text
video_shape: (1, 16, 3, 224, 224)
gazing_pos_shape: (1, 348)
if_padded_gazing_shape: (1, 348)
num_non_padded_gazes: 213
num_gazing_each_frame: [198, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
scales: [32, 64, 112, 224]
num_vision_tokens_each_frame: 265
```

## 6. Run AutoGaze With The Custom SigLIP Encoder

The repo includes a SigLIP implementation that only processes the gazed patches. This is the tested path:

```bash
MAMBA_ROOT_PREFIX="$HOME/.micromamba" \
"$HOME/.local/bin/micromamba" run -p "$HOME/.micromamba/envs/autogaze" \
python - <<'PY'
import av
import torch
from transformers import AutoImageProcessor
from autogaze.datasets.video_utils import read_video_pyav, transform_video_for_pytorch
from autogaze.models.autogaze import AutoGazeImageProcessor, AutoGaze
from autogaze.vision_encoders.siglip import SiglipVisionModel

video_path = "assets/example_input.mp4"
device = "cuda"
autogaze_revision = "5100fae739ec1bf3f875914fa1b703846a18943a"
siglip_revision = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"

autogaze_transform = AutoGazeImageProcessor.from_pretrained("nvidia/AutoGaze", revision=autogaze_revision)
autogaze_model = AutoGaze.from_pretrained("nvidia/AutoGaze", revision=autogaze_revision).to(device).eval()

siglip_transform = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224", revision=siglip_revision)
siglip_model = SiglipVisionModel.from_pretrained(
    "google/siglip2-base-patch16-224",
    revision=siglip_revision,
    scales=autogaze_model.config.scales,
    attn_implementation="sdpa",
).to(device).eval()

container = av.open(video_path)
raw_video = read_video_pyav(container, list(range(16)))
container.close()

video_input_autogaze = transform_video_for_pytorch(raw_video, autogaze_transform)[None].to(device)
video_input_siglip = transform_video_for_pytorch(raw_video, siglip_transform)[None].to(device)

with torch.inference_mode():
    gaze_outputs = autogaze_model({"video": video_input_autogaze}, gazing_ratio=0.75, task_loss_requirement=0.7)
    siglip_outputs = siglip_model(video_input_siglip, gazing_info=gaze_outputs)

features = [feat[~if_pad] for feat, if_pad in zip(siglip_outputs.last_hidden_state, gaze_outputs["if_padded_gazing"])]

print("siglip_output_shape:", tuple(siglip_outputs.last_hidden_state.shape))
print("non_padded_feature_shape:", tuple(features[0].shape))
print("num_non_padded:", int((~gaze_outputs["if_padded_gazing"]).sum().item()))
PY
```

Tested output:

```text
siglip_output_shape: (1, 348, 768)
non_padded_feature_shape: (213, 768)
num_non_padded: 213
```

## 7. Run The Systematic Reconstruction Evaluation

This fork adds [`scripts/evaluate_reconstruction.py`](scripts/evaluate_reconstruction.py), which evaluates AutoGaze end-to-end on real clips and writes plots/metrics for different gaze budgets and loss thresholds.

### What the script does

- Loads `nvidia/AutoGaze`
- Loads the released reconstruction task checkpoint from `bfshi/VideoMAE_AutoGaze`
- Splits an input video into 16-frame clips
- Evaluates these settings:
  - `ratio_0.10`
  - `ratio_0.25`
  - `ratio_0.50`
  - `ratio_0.75`
  - `ratio_0.75_task_0.70`
  - `task_0.70`
  - `ratio_1.00`
- Computes:
  - actual gaze ratio
  - total non-padded gazes
  - reconstruction loss
  - pixel MAE
  - pixel MSE
  - PSNR
  - SSIM
  - gaze runtime
  - reconstruction runtime
- Saves:
  - CSV tables
  - text summary
  - tradeoff plots
  - per-frame plots
  - a gaze heatmap
  - qualitative image grids

### Run it on the bundled example video

The evaluator defaults are already pinned to the exact tested Hugging Face revisions listed above, so the shortest correct command is:

```bash
MAMBA_ROOT_PREFIX="$HOME/.micromamba" \
"$HOME/.local/bin/micromamba" run -p "$HOME/.micromamba/envs/autogaze" \
python scripts/evaluate_reconstruction.py
```

### Run it on your own video

```bash
MAMBA_ROOT_PREFIX="$HOME/.micromamba" \
"$HOME/.local/bin/micromamba" run -p "$HOME/.micromamba/envs/autogaze" \
python scripts/evaluate_reconstruction.py \
  --video /absolute/path/to/video.mp4 \
  --output-dir analysis_outputs/my_video_eval \
  --clip-len 16 \
  --max-clips 8 \
  --viz-clips 2
```

If you want the command line itself to show the exact pinned revisions rather than relying on defaults:

```bash
MAMBA_ROOT_PREFIX="$HOME/.micromamba" \
"$HOME/.local/bin/micromamba" run -p "$HOME/.micromamba/envs/autogaze" \
python scripts/evaluate_reconstruction.py \
  --autogaze-model nvidia/AutoGaze \
  --autogaze-revision 5100fae739ec1bf3f875914fa1b703846a18943a \
  --task-asset-repo bfshi/VideoMAE_AutoGaze \
  --task-asset-revision 34001937344859e687d336cd44ec8962018ae46b
```

### Important arguments

- `--video`: video file to evaluate
- `--output-dir`: where artifacts are written
- `--task-asset-dir`: cache directory for the reconstruction checkpoint
- `--autogaze-revision`: pinned AutoGaze revision, defaulting to the exact tested SHA
- `--task-asset-repo`: released VideoMAE_AutoGaze repo
- `--task-asset-revision`: pinned VideoMAE_AutoGaze revision, defaulting to the exact tested SHA
- `--clip-len`: frames per clip, default `16`
- `--clip-starts`: comma-separated start indices, for explicit clip selection
- `--max-clips`: maximum number of clips to evaluate
- `--viz-clips`: how many clips get qualitative PNG grids
- `--attn-mode`: `flash_attention_2`, `sdpa`, or `eager`
- `--device`: currently expected to be `cuda`

### Output layout

The default output directory is:

```text
analysis_outputs/reconstruction_eval/
├── metadata.json
├── metrics_by_clip_setting.csv
├── metrics_by_frame.csv
├── plots/
│   ├── gaze_heatmap.png
│   ├── per_frame_curves.png
│   └── tradeoff_curves.png
├── summary.txt
├── summary_by_setting.csv
└── visualizations/
    └── clip_00_start_0000_*.png
```

## 8. Example Results On The Bundled Demo Video

The checked-in artifacts under [`analysis_outputs/reconstruction_eval`](analysis_outputs/reconstruction_eval) were produced from the repository's bundled `assets/example_input.mp4`, using four 16-frame clips starting at `0, 16, 32, 48`.

Summary:

| Setting | Actual Gaze Ratio | Non-Padded Gazes | Reconstruction Loss | Pixel MAE | PSNR | SSIM | Gaze Seconds | Reconstruction Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ratio_0.75_task_0.70` | 0.0528 | 223.8 | 0.9398 | 0.0251 | 26.809 | 0.8875 | 0.2738 | 0.1158 |
| `task_0.70` | 0.0683 | 289.8 | 0.8067 | 0.0238 | 27.130 | 0.9065 | 0.3369 | 0.1203 |
| `ratio_0.10` | 0.0981 | 416.0 | 1.2928 | 0.0298 | 25.056 | 0.8384 | 0.4687 | 0.1374 |
| `ratio_0.25` | 0.2491 | 1056.0 | 0.9254 | 0.0218 | 27.753 | 0.9032 | 0.7907 | 0.1131 |
| `ratio_0.50` | 0.4981 | 2112.0 | 0.8093 | 0.0193 | 28.998 | 0.9214 | 1.4913 | 0.1162 |
| `ratio_0.75` | 0.7472 | 3168.0 | 0.7019 | 0.0179 | 29.549 | 0.9335 | 2.1793 | 0.1328 |
| `ratio_1.00` | 1.0000 | 4240.0 | 0.7174 | 0.0189 | 29.380 | 0.9270 | 3.1802 | 0.1620 |

Observed behavior on this video:

- `ratio_0.75` produced the best quality among the tested fixed-ratio settings.
- `ratio_1.00` was slightly worse than `ratio_0.75` on these clips, so more patches were not automatically better.
- `task_0.70` was the best quality/efficiency operating point in this sweep, reaching strong reconstruction quality with only about `6.8%` actual gaze ratio.
- The thresholded modes concentrated almost all gazes on the first frame, then used only about `1-5` patches on most later frames.

See:

- [`analysis_outputs/reconstruction_eval/summary.txt`](analysis_outputs/reconstruction_eval/summary.txt)
- [`analysis_outputs/reconstruction_eval/plots/tradeoff_curves.png`](analysis_outputs/reconstruction_eval/plots/tradeoff_curves.png)
- [`analysis_outputs/reconstruction_eval/plots/per_frame_curves.png`](analysis_outputs/reconstruction_eval/plots/per_frame_curves.png)
- [`analysis_outputs/reconstruction_eval/plots/gaze_heatmap.png`](analysis_outputs/reconstruction_eval/plots/gaze_heatmap.png)

### Expected Warnings During Evaluation

The following warnings are expected on the tested setup and do not indicate a broken run:

- the temporary warning that some `ViTMAEForPreTraining` weights are newly initialized from `facebook/vit-mae-large` before `videomae.pt` is loaded,
- the `use_fast` image processor warning from Hugging Face,
- the FlashAttention initialization warning about torch dtype when the task model is first built.

## 9. Training

Training is unchanged from the upstream release. See [`TRAIN.md`](TRAIN.md) for:

- dataset download
- expected folder layout
- NTP pretraining
- RL post-training
- all relevant Hydra config knobs

## 10. Troubleshooting

### `transformers` import or model errors

Use `transformers 4.51.x`. This repository now pins:

```text
transformers>=4.51,<4.52
```

Later releases can move or rename internal HF modules used by this codebase.

### `flash-attn` installation failures

If `pip install flash-attn==2.8.3` tries to build from source and fails, install the matching prebuilt wheel directly as shown above. The tested wheel was:

```text
flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
```

### CUDA is available in `nvidia-smi` but not in PyTorch

Check:

- the environment is actually being used
- PyTorch was installed from the CUDA 12.8 wheel index
- `torch.cuda.is_available()` returns `True`

### A fresh run downloads different model files than expected

Use the pinned revision-aware commands in this README or the default evaluator arguments. For exact isolation, keep:

```bash
export HF_HOME="$PWD/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
```

before running any `from_pretrained(...)` code.

### Evaluation script fails on CPU-only machines

That is expected. [`scripts/evaluate_reconstruction.py`](scripts/evaluate_reconstruction.py) currently assumes CUDA.

### The reconstruction checkpoint appears mismatched

The released `videomae.pt` checkpoint stores keys with a `module.` prefix from DDP training. The evaluator strips that prefix automatically before loading.

### `pytest` reports no tests

This repository currently does not ship unit tests. On the tested setup:

```text
collected 0 items
```

For practical validation, use:

- the AutoGaze inference smoke test
- the SigLIP integration smoke test
- the reconstruction evaluation script

### Metric values differ slightly from the checked-in tables

On the same machine class and with the exact commands above, the reported metrics should match to the displayed precision. Small floating-point drift remains possible across different GPUs, drivers, or CUDA library stacks.

## 11. Code Structure

The main package layout is:

```text
autogaze/
├── algorithms/
├── configs/
├── datasets/
├── models/
│   └── autogaze/
├── tasks/
│   └── video_mae_reconstruction/
├── vision_encoders/
│   └── siglip/
├── train.py
└── trainer.py
```

Component summary:

- `models`: AutoGaze and related gaze-model code.
- `tasks`: task models and losses, including VideoMAE reconstruction.
- `algorithms`: training algorithms such as NTP and GRPO.
- `datasets`: video loading, frame sampling, and collation.
- `vision_encoders`: vision backbones adapted to sparse gazed patches.
- `configs`: Hydra configuration trees for training and evaluation-related settings.

## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{shi2026attendattentionefficientscalable,
      title={Attend Before Attention: Efficient and Scalable Video Understanding via Autoregressive Gazing},
      author={Baifeng Shi and Stephanie Fu and Long Lian and Hanrong Ye and David Eigen and Aaron Reite and Boyi Li and Jan Kautz and Song Han and David M. Chan and Pavlo Molchanov and Trevor Darrell and Hongxu Yin},
      year={2026},
      eprint={2603.12254},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.12254},
}
```
