#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path

import av
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from autogaze.datasets.video_utils import process_video_frames, read_video_pyav, transform_video_for_pytorch
from autogaze.models.autogaze import AutoGaze, AutoGazeImageProcessor
from autogaze.tasks.video_mae_reconstruction import VideoMAEReconstruction
from autogaze.utils import UnNormalize


PINNED_MODEL_REVISIONS = {
    "nvidia/AutoGaze": "5100fae739ec1bf3f875914fa1b703846a18943a",
    "bfshi/VideoMAE_AutoGaze": "34001937344859e687d336cd44ec8962018ae46b",
    "facebook/vit-mae-large": "142cb8c25e1b1bc1769997a919aa1b5a2345a6b8",
    "facebook/dinov2-with-registers-base": "a1d738ccfa7ae170945f210395d99dde8adb1805",
    "google/siglip2-base-patch16-224": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
}


DEFAULT_SETTINGS = [
    {"name": "ratio_0.10", "gazing_ratio": 0.10, "task_loss_requirement": None},
    {"name": "ratio_0.25", "gazing_ratio": 0.25, "task_loss_requirement": None},
    {"name": "ratio_0.50", "gazing_ratio": 0.50, "task_loss_requirement": None},
    {"name": "ratio_0.75", "gazing_ratio": 0.75, "task_loss_requirement": None},
    {"name": "ratio_0.75_task_0.70", "gazing_ratio": 0.75, "task_loss_requirement": 0.70},
    {"name": "task_0.70", "gazing_ratio": None, "task_loss_requirement": 0.70},
    {"name": "ratio_1.00", "gazing_ratio": 1.00, "task_loss_requirement": None},
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run a systematic AutoGaze reconstruction evaluation.")
    parser.add_argument(
        "--video",
        type=Path,
        default=repo_root / "assets" / "example_input.mp4",
        help="Video to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "analysis_outputs" / "reconstruction_eval",
        help="Directory for metrics, plots, and visualizations.",
    )
    parser.add_argument(
        "--task-asset-dir",
        type=Path,
        default=repo_root / "hf_cache" / "VideoMAE_AutoGaze",
        help="Directory used to cache the VideoMAE_AutoGaze task assets.",
    )
    parser.add_argument(
        "--autogaze-model",
        type=str,
        default="nvidia/AutoGaze",
        help="Hugging Face repo for the AutoGaze model.",
    )
    parser.add_argument(
        "--autogaze-revision",
        type=str,
        default=PINNED_MODEL_REVISIONS["nvidia/AutoGaze"],
        help="Pinned Hugging Face revision for the AutoGaze model.",
    )
    parser.add_argument(
        "--task-asset-repo",
        type=str,
        default="bfshi/VideoMAE_AutoGaze",
        help="Hugging Face repo for the released VideoMAE_AutoGaze task assets.",
    )
    parser.add_argument(
        "--task-asset-revision",
        type=str,
        default=PINNED_MODEL_REVISIONS["bfshi/VideoMAE_AutoGaze"],
        help="Pinned Hugging Face revision for the VideoMAE_AutoGaze task assets.",
    )
    parser.add_argument(
        "--clip-len",
        type=int,
        default=16,
        help="Number of frames per clip.",
    )
    parser.add_argument(
        "--clip-starts",
        type=str,
        default="",
        help="Comma-separated frame indices for clip starts. Defaults to non-overlapping clips.",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=4,
        help="Maximum number of clips to evaluate when --clip-starts is not provided.",
    )
    parser.add_argument(
        "--viz-clips",
        type=int,
        default=1,
        help="Number of clips to save qualitative visualization grids for.",
    )
    parser.add_argument(
        "--attn-mode",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention implementation for the feature-loss backbones in the reconstruction task.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for any downstream sampling.",
    )
    return parser.parse_args()


def get_video_num_frames(video_path: Path) -> int:
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    num_frames = int(stream.frames or 0)
    if num_frames == 0:
        num_frames = sum(1 for _ in container.decode(video=0))
    container.close()
    return num_frames


def get_clip_starts(num_frames: int, clip_len: int, clip_starts_arg: str, max_clips: int) -> list[int]:
    if clip_starts_arg:
        starts = [int(x.strip()) for x in clip_starts_arg.split(",") if x.strip()]
    else:
        starts = list(range(0, max(num_frames, 1), clip_len))
    starts = [start for start in starts if start < num_frames]
    if not starts:
        starts = [0]
    return starts[:max_clips]


def load_clip(video_path: Path, start: int, clip_len: int) -> np.ndarray:
    container = av.open(str(video_path))
    indices = list(range(start, start + clip_len))
    raw_video = read_video_pyav(container, indices)
    container.close()
    return process_video_frames(raw_video, clip_len)


def download_task_assets(task_asset_repo: str, task_asset_revision: str, local_dir: Path) -> tuple[Path, Path]:
    local_dir.mkdir(parents=True, exist_ok=True)
    config_path = hf_hub_download(
        task_asset_repo,
        "config.yaml",
        revision=task_asset_revision,
        local_dir=str(local_dir),
    )
    weights_path = hf_hub_download(
        task_asset_repo,
        "videomae.pt",
        revision=task_asset_revision,
        local_dir=str(local_dir),
    )
    return Path(config_path), Path(weights_path)


def load_task(args: argparse.Namespace) -> VideoMAEReconstruction:
    config_path, weights_path = download_task_assets(
        args.task_asset_repo,
        args.task_asset_revision,
        args.task_asset_dir,
    )
    cfg = OmegaConf.load(config_path)
    task_cfg = cfg.task
    task_cfg.recon_model_config.revision = PINNED_MODEL_REVISIONS[task_cfg.recon_model]
    task_cfg.recon_model_config.dinov2_reg_loss_config.revision = PINNED_MODEL_REVISIONS[
        task_cfg.recon_model_config.dinov2_reg_loss_config.model
    ]
    task_cfg.recon_model_config.siglip2_loss_config.revision = PINNED_MODEL_REVISIONS[
        task_cfg.recon_model_config.siglip2_loss_config.model
    ]
    task = VideoMAEReconstruction(
        recon_model=task_cfg.recon_model,
        recon_model_config=task_cfg.recon_model_config,
        scales=task_cfg.scales,
        recon_sample_rate=1,
        attn_mode=args.attn_mode,
    ).to(args.device).eval()
    state = torch.load(weights_path, map_location="cpu")
    state = OrderedDict((key.removeprefix("module."), value) for key, value in state.items())
    missing_keys, unexpected_keys = task.load_state_dict(state, strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Task checkpoint did not align with the current code. "
            f"Missing: {missing_keys[:5]}, unexpected: {unexpected_keys[:5]}"
        )
    return task


def load_models(args: argparse.Namespace) -> tuple[AutoGazeImageProcessor, AutoGaze, VideoMAEReconstruction]:
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    processor = AutoGazeImageProcessor.from_pretrained(args.autogaze_model, revision=args.autogaze_revision)
    gaze_model = AutoGaze.from_pretrained(args.autogaze_model, revision=args.autogaze_revision).to(args.device).eval()
    task = load_task(args)
    return processor, gaze_model, task


def get_non_padded_gazes_each_frame(gaze_outputs: dict) -> list[int]:
    num_gazing_each_frame = gaze_outputs["num_gazing_each_frame"].detach().cpu().tolist()
    if_padded = gaze_outputs["if_padded_gazing"][0].detach().cpu()
    counts = []
    start = 0
    for frame_count in num_gazing_each_frame:
        end = start + frame_count
        counts.append(int((~if_padded[start:end]).sum().item()))
        start = end
    return counts


def unnormalize_video(video: torch.Tensor, image_mean, image_std, rescale_factor) -> torch.Tensor:
    return UnNormalize(image_mean, image_std, rescale_factor)(video).detach().cpu().float()


def compute_frame_metrics(original: torch.Tensor, reconstruction: torch.Tensor) -> list[dict]:
    original_np = original.permute(0, 2, 3, 1).numpy()
    reconstruction_np = reconstruction.permute(0, 2, 3, 1).numpy()
    rows = []
    for frame_idx, (orig_frame, recon_frame) in enumerate(zip(original_np, reconstruction_np)):
        pixel_mae = float(np.abs(orig_frame - recon_frame).mean())
        pixel_mse = float(np.mean((orig_frame - recon_frame) ** 2))
        psnr = float(peak_signal_noise_ratio(orig_frame, recon_frame, data_range=1.0))
        ssim = float(structural_similarity(orig_frame, recon_frame, channel_axis=-1, data_range=1.0))
        rows.append(
            {
                "frame_idx": frame_idx,
                "pixel_mae": pixel_mae,
                "pixel_mse": pixel_mse,
                "psnr": psnr,
                "ssim": ssim,
            }
        )
    return rows


def save_visualization_grid(
    clip_name: str,
    setting_name: str,
    original: torch.Tensor,
    reconstruction: torch.Tensor,
    gaze_outputs: dict,
    scales: list[int],
    output_path: Path,
) -> None:
    num_scales = len(scales)
    num_frames = original.shape[0]
    fig, axes = plt.subplots(num_scales + 2, num_frames, figsize=(2.2 * num_frames, 2.2 * (num_scales + 2)))
    if num_frames == 1:
        axes = np.expand_dims(axes, axis=1)

    original_np = original.numpy()
    reconstruction_np = reconstruction.numpy()
    gazing_mask = [mask[0].detach().cpu() for mask in gaze_outputs["gazing_mask"]]

    for frame_idx in range(num_frames):
        axes[0, frame_idx].imshow(original_np[frame_idx].transpose(1, 2, 0))
        axes[0, frame_idx].set_title(f"Orig {frame_idx}")
        axes[0, frame_idx].axis("off")

    for scale_idx, scale in enumerate(scales):
        scale_mask = gazing_mask[scale_idx]
        for frame_idx in range(num_frames):
            frame_mask = scale_mask[frame_idx]
            grid_size = int(frame_mask.numel() ** 0.5)
            frame_mask = frame_mask.reshape(grid_size, grid_size)
            frame_mask_resized = F.interpolate(
                frame_mask[None, None].float(),
                size=(scale, scale),
                mode="nearest",
            )[0, 0].numpy()
            frame_tensor = torch.from_numpy(original_np[frame_idx]).unsqueeze(0)
            scaled_frame = (
                F.interpolate(frame_tensor, size=(scale, scale), mode="bicubic", align_corners=False)[0]
                .clamp(0, 1)
                .numpy()
            )
            masked_frame = scaled_frame * (0.8 * frame_mask_resized[None] + 0.2)
            axes[scale_idx + 1, frame_idx].imshow(masked_frame.transpose(1, 2, 0))

            patch_size = scale // grid_size
            for row_idx in range(grid_size):
                for col_idx in range(grid_size):
                    if frame_mask[row_idx, col_idx] > 0.5:
                        rect = plt.Rectangle(
                            (col_idx * patch_size - 0.5, row_idx * patch_size - 0.5),
                            patch_size,
                            patch_size,
                            linewidth=0.8,
                            edgecolor="red",
                            facecolor="none",
                        )
                        axes[scale_idx + 1, frame_idx].add_patch(rect)

            axes[scale_idx + 1, frame_idx].set_title(f"{scale}px")
            axes[scale_idx + 1, frame_idx].axis("off")

    for frame_idx in range(num_frames):
        axes[num_scales + 1, frame_idx].imshow(reconstruction_np[frame_idx].transpose(1, 2, 0))
        axes[num_scales + 1, frame_idx].set_title(f"Recon {frame_idx}")
        axes[num_scales + 1, frame_idx].axis("off")

    fig.suptitle(f"{clip_name} | {setting_name}", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def annotate_points(ax: plt.Axes, data: pd.DataFrame, x: str, y: str, label: str) -> None:
    for _, row in data.iterrows():
        ax.annotate(row[label], (row[x], row[y]), xytext=(5, 5), textcoords="offset points", fontsize=8)


def plot_tradeoffs(summary_df: pd.DataFrame, plot_dir: Path) -> None:
    ordered = summary_df.sort_values("actual_gaze_ratio")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.lineplot(data=ordered, x="actual_gaze_ratio", y="reconstruction_loss", marker="o", ax=axes[0, 0])
    axes[0, 0].set_title("Reconstruction Loss vs Actual Gaze Ratio")
    annotate_points(axes[0, 0], ordered, "actual_gaze_ratio", "reconstruction_loss", "setting")

    sns.lineplot(data=ordered, x="actual_gaze_ratio", y="pixel_mae", marker="o", ax=axes[0, 1])
    axes[0, 1].set_title("Pixel MAE vs Actual Gaze Ratio")
    annotate_points(axes[0, 1], ordered, "actual_gaze_ratio", "pixel_mae", "setting")

    sns.lineplot(data=ordered, x="actual_gaze_ratio", y="psnr", marker="o", ax=axes[1, 0])
    axes[1, 0].set_title("PSNR vs Actual Gaze Ratio")
    annotate_points(axes[1, 0], ordered, "actual_gaze_ratio", "psnr", "setting")

    sns.lineplot(data=ordered, x="actual_gaze_ratio", y="ssim", marker="o", ax=axes[1, 1])
    axes[1, 1].set_title("SSIM vs Actual Gaze Ratio")
    annotate_points(axes[1, 1], ordered, "actual_gaze_ratio", "ssim", "setting")

    for ax in axes.flat:
        ax.set_xlabel("Actual Gaze Ratio")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(plot_dir / "tradeoff_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_per_frame_curves(frame_df: pd.DataFrame, plot_dir: Path) -> None:
    frame_summary = (
        frame_df.groupby(["setting", "frame_idx"], as_index=False)
        .agg(
            reconstruction_loss=("reconstruction_loss", "mean"),
            non_padded_gazes=("non_padded_gazes", "mean"),
            psnr=("psnr", "mean"),
            ssim=("ssim", "mean"),
        )
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    sns.lineplot(data=frame_summary, x="frame_idx", y="reconstruction_loss", hue="setting", marker="o", ax=axes[0])
    axes[0].set_title("Per-Frame Reconstruction Loss")
    axes[0].grid(True, alpha=0.3)

    sns.lineplot(data=frame_summary, x="frame_idx", y="non_padded_gazes", hue="setting", marker="o", ax=axes[1])
    axes[1].set_title("Per-Frame Non-Padded Gaze Count")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel("Frame Index")

    plt.tight_layout()
    fig.savefig(plot_dir / "per_frame_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    heatmap_df = frame_summary.pivot(index="setting", columns="frame_idx", values="non_padded_gazes")
    plt.figure(figsize=(14, 4.5))
    sns.heatmap(heatmap_df, annot=True, fmt=".1f", cmap="mako")
    plt.title("Mean Gaze Count Per Frame")
    plt.xlabel("Frame Index")
    plt.ylabel("Setting")
    plt.tight_layout()
    plt.savefig(plot_dir / "gaze_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close()


def write_summary(summary_df: pd.DataFrame, clip_starts: list[int], output_path: Path) -> None:
    ordered = summary_df.sort_values("actual_gaze_ratio").copy()
    ordered = ordered[
        [
            "setting",
            "actual_gaze_ratio",
            "non_padded_gazes",
            "reconstruction_loss",
            "pixel_mae",
            "psnr",
            "ssim",
            "gaze_seconds",
            "reconstruction_seconds",
        ]
    ]
    ordered = ordered.round(
        {
            "actual_gaze_ratio": 4,
            "non_padded_gazes": 1,
            "reconstruction_loss": 4,
            "pixel_mae": 4,
            "psnr": 3,
            "ssim": 4,
            "gaze_seconds": 4,
            "reconstruction_seconds": 4,
        }
    )

    lines = [
        "# AutoGaze Reconstruction Evaluation",
        "",
        f"- Clip starts: {clip_starts}",
        f"- Number of evaluated clips: {len(clip_starts)}",
        "",
        "## Mean Metrics by Setting",
        "",
        ordered.to_string(index=False),
        "",
    ]
    output_path.write_text("\n".join(lines))


def evaluate(args: argparse.Namespace) -> None:
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This evaluation script expects a working CUDA device.")

    sns.set_theme(style="whitegrid")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output_dir / "plots"
    viz_dir = args.output_dir / "visualizations"
    plot_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    processor, gaze_model, task = load_models(args)

    num_frames = get_video_num_frames(args.video)
    clip_starts = get_clip_starts(num_frames, args.clip_len, args.clip_starts, args.max_clips)
    settings = list(DEFAULT_SETTINGS)

    clip_rows = []
    frame_rows = []
    metadata_rows = []

    for clip_idx, clip_start in enumerate(clip_starts):
        clip_name = f"clip_{clip_idx:02d}_start_{clip_start:04d}"
        raw_clip = load_clip(args.video, clip_start, args.clip_len)
        video_gaze = transform_video_for_pytorch(raw_clip, processor)[None].to(args.device)
        video_task = transform_video_for_pytorch(raw_clip, task.transform)[None].to(args.device)
        inputs = {"video": video_gaze, "video_for_task": video_task}

        for setting in settings:
            model_kwargs = {}
            if setting["gazing_ratio"] is not None:
                model_kwargs["gazing_ratio"] = setting["gazing_ratio"]
            if setting["task_loss_requirement"] is not None:
                model_kwargs["task_loss_requirement"] = setting["task_loss_requirement"]

            with torch.inference_mode():
                torch.cuda.synchronize()
                gaze_start = time.perf_counter()
                gaze_outputs = gaze_model({"video": video_gaze}, **model_kwargs)
                torch.cuda.synchronize()
                gaze_seconds = time.perf_counter() - gaze_start

                torch.cuda.synchronize()
                recon_start = time.perf_counter()
                task_outputs = task.forward_output(
                    inputs,
                    gaze_outputs,
                    frame_idx_to_reconstruct=torch.arange(video_task.shape[1], device=video_task.device),
                )
                torch.cuda.synchronize()
                reconstruction_seconds = time.perf_counter() - recon_start

            original = unnormalize_video(
                video_task[0],
                task_outputs["image_mean"],
                task_outputs["image_std"],
                task_outputs["rescale_factor"],
            )
            reconstruction = unnormalize_video(
                task_outputs["reconstruction"][0],
                task_outputs["image_mean"],
                task_outputs["image_std"],
                task_outputs["rescale_factor"],
            )

            frame_metrics = compute_frame_metrics(original, reconstruction)
            non_padded_gazes_by_frame = get_non_padded_gazes_each_frame(gaze_outputs)
            frame_reconstruction_losses = (
                task_outputs["reconstruction_loss_each_reconstruction_frame"][0].detach().cpu().float().tolist()
            )
            non_padded_gazes = int((~gaze_outputs["if_padded_gazing"]).sum().item())
            actual_gaze_ratio = non_padded_gazes / (
                video_task.shape[1] * int(gaze_outputs["num_vision_tokens_each_frame"])
            )

            clip_row = {
                "clip_name": clip_name,
                "clip_start": clip_start,
                "setting": setting["name"],
                "gazing_ratio_arg": setting["gazing_ratio"],
                "task_loss_requirement_arg": setting["task_loss_requirement"],
                "non_padded_gazes": non_padded_gazes,
                "actual_gaze_ratio": actual_gaze_ratio,
                "reconstruction_loss": float(task_outputs["reconstruction_loss"].item()),
                "pixel_mae": float(np.mean([row["pixel_mae"] for row in frame_metrics])),
                "pixel_mse": float(np.mean([row["pixel_mse"] for row in frame_metrics])),
                "psnr": float(np.mean([row["psnr"] for row in frame_metrics])),
                "ssim": float(np.mean([row["ssim"] for row in frame_metrics])),
                "gaze_seconds": gaze_seconds,
                "reconstruction_seconds": reconstruction_seconds,
                "non_padded_gazes_by_frame": json.dumps(non_padded_gazes_by_frame),
                "num_gazing_each_frame": json.dumps(gaze_outputs["num_gazing_each_frame"].detach().cpu().tolist()),
            }
            clip_rows.append(clip_row)

            for frame_metric, frame_reconstruction_loss, non_padded_frame_gazes in zip(
                frame_metrics,
                frame_reconstruction_losses,
                non_padded_gazes_by_frame,
            ):
                frame_rows.append(
                    {
                        "clip_name": clip_name,
                        "clip_start": clip_start,
                        "setting": setting["name"],
                        "frame_idx": frame_metric["frame_idx"],
                        "reconstruction_loss": frame_reconstruction_loss,
                        "non_padded_gazes": non_padded_frame_gazes,
                        "pixel_mae": frame_metric["pixel_mae"],
                        "pixel_mse": frame_metric["pixel_mse"],
                        "psnr": frame_metric["psnr"],
                        "ssim": frame_metric["ssim"],
                    }
                )

            metadata_rows.append(
                {
                    "clip_name": clip_name,
                    "setting": setting["name"],
                    "gaze_outputs": {
                        "scales": gaze_outputs["scales"],
                        "num_vision_tokens_each_frame": int(gaze_outputs["num_vision_tokens_each_frame"]),
                        "task_loss_requirement": (
                            None
                            if gaze_outputs["task_loss_requirement"] is None
                            else gaze_outputs["task_loss_requirement"].detach().cpu().tolist()
                        ),
                    },
                }
            )

            if clip_idx < args.viz_clips:
                save_visualization_grid(
                    clip_name=clip_name,
                    setting_name=setting["name"],
                    original=original,
                    reconstruction=reconstruction,
                    gaze_outputs=gaze_outputs,
                    scales=task_outputs["scales"],
                    output_path=viz_dir / f"{clip_name}_{setting['name']}.png",
                )

    clip_df = pd.DataFrame(clip_rows)
    frame_df = pd.DataFrame(frame_rows)
    summary_df = (
        clip_df.groupby("setting", as_index=False)
        .agg(
            actual_gaze_ratio=("actual_gaze_ratio", "mean"),
            non_padded_gazes=("non_padded_gazes", "mean"),
            reconstruction_loss=("reconstruction_loss", "mean"),
            pixel_mae=("pixel_mae", "mean"),
            pixel_mse=("pixel_mse", "mean"),
            psnr=("psnr", "mean"),
            ssim=("ssim", "mean"),
            gaze_seconds=("gaze_seconds", "mean"),
            reconstruction_seconds=("reconstruction_seconds", "mean"),
        )
    )

    clip_df.to_csv(args.output_dir / "metrics_by_clip_setting.csv", index=False)
    frame_df.to_csv(args.output_dir / "metrics_by_frame.csv", index=False)
    summary_df.to_csv(args.output_dir / "summary_by_setting.csv", index=False)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata_rows, indent=2))

    plot_tradeoffs(summary_df, plot_dir)
    plot_per_frame_curves(frame_df, plot_dir)
    write_summary(summary_df, clip_starts, args.output_dir / "summary.txt")

    print(f"Evaluated {len(clip_starts)} clips across {len(settings)} settings.")
    print(summary_df.sort_values('actual_gaze_ratio').round(4).to_string(index=False))
    print(f"Artifacts written to: {args.output_dir}")


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
