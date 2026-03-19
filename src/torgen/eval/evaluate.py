"""GPU evaluation pipeline for TorGen v2.

Generates multiple z samples per test day, collects tracks into a DataFrame,
computes distributional metrics, and saves results.
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import stats

from torgen.data.dataset import TornadoDataset
from torgen.training.config import TrainConfig


def run_evaluation(
    model: torch.nn.Module,
    cfg: TrainConfig,
    data_dir: str,
    output_dir: str,
    split: str | None = "test",
    device: torch.device | None = None,
    chunk_size: int = 25,
) -> dict[str, Any]:
    """Run full evaluation pipeline.

    For each day in the dataset, generates n_eval_samples realizations,
    collects predicted tracks, computes metrics, and saves outputs.

    Args:
        model: Trained TorGenCVAE.
        cfg: Training config (uses n_eval_samples, exists_threshold).
        data_dir: Path to .pt files.
        output_dir: Where to save samples.parquet, summary.json, plots.
        split: Dataset split to evaluate ("train", "val", "test", or None for all).
        device: Torch device. Defaults to model's device.
        chunk_size: Batch size for z sampling to avoid OOM.

    Returns:
        Summary metrics dict.
    """
    if device is None:
        device = next(model.parameters()).device

    os.makedirs(output_dir, exist_ok=True)

    ds = TornadoDataset(data_dir, preload=False, split=split)
    if len(ds) == 0:
        summary = {"count_mae": 0.0, "count_ks_pvalue": 1.0, "n_days": 0}
        with open(os.path.join(output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        pd.DataFrame().to_parquet(os.path.join(output_dir, "samples.parquet"))
        return summary

    n_samples = cfg.n_eval_samples
    threshold = cfg.exists_threshold

    model.eval()

    # Weather channel indices for environment summary
    _WX_CHANNELS = {"stp": 8, "scp": 9, "acpcp": 5}

    all_rows: list[dict[str, Any]] = []
    gt_rows: list[dict[str, Any]] = []
    env_rows: list[dict[str, Any]] = []
    exists_rows: list[dict[str, Any]] = []
    gt_counts: list[int] = []
    gen_counts_per_day: list[list[int]] = []

    for day_idx in range(len(ds)):
        sample = ds[day_idx]
        date = sample.get("date", f"day_{day_idx}")
        gt_tracks = sample["tracks"]  # (N, 6) in bearing/length format
        n_gt = gt_tracks.shape[0]
        gt_counts.append(n_gt)

        for i in range(n_gt):
            t = gt_tracks[i]
            gt_rows.append({
                "date": date,
                "se": t[0].item(), "sn": t[1].item(),
                "bearing": t[2].item(), "length": t[3].item(),
                "width": t[4].item(), "ef": int(t[5].item()),
            })

        # Environment summary (p99 of key channels)
        wx_raw = sample["wx"]  # (C, H, W)
        env_row: dict[str, Any] = {"date": date}
        for name, ch in _WX_CHANNELS.items():
            if ch < wx_raw.shape[0]:
                env_row[f"p99_{name}"] = float(wx_raw[ch].quantile(0.99))
        env_rows.append(env_row)

        wx = wx_raw.unsqueeze(0).to(device)  # (1, C, H, W)

        day_gen_counts: list[int] = []

        for chunk_start in range(0, n_samples, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_samples)
            n_chunk = chunk_end - chunk_start

            wx_batch = wx.expand(n_chunk, -1, -1, -1)

            with torch.no_grad():
                out = model.generate(wx_batch)

            preds = out["preds"]
            exists = preds["exists"]          # (n_chunk, Q, 1)
            coords = preds["coords"]          # (n_chunk, Q, 2)
            bearing = preds["bearing"]        # (n_chunk, Q, 1)
            length = preds["length"]          # (n_chunk, Q, 1)
            width = preds["width"]            # (n_chunk, Q, 1)
            ef_logits = preds["ef_logits"]    # (n_chunk, Q, n_ef)

            ef_class = ef_logits.argmax(dim=-1)  # (n_chunk, Q)

            for i in range(n_chunk):
                realization_id = chunk_start + i
                exists_probs = exists[i].squeeze(-1)  # (Q,)
                mask = exists_probs > threshold  # (Q,)
                n_pred = mask.sum().item()
                day_gen_counts.append(n_pred)

                # Save ALL slot exists probs
                probs_np = exists_probs.cpu().numpy()
                for slot_idx in range(probs_np.shape[0]):
                    exists_rows.append({
                        "date": date,
                        "realization_id": realization_id,
                        "slot": slot_idx,
                        "exists_prob": float(probs_np[slot_idx]),
                    })

                if n_pred > 0:
                    idx = mask.nonzero(as_tuple=True)[0]
                    for j in idx:
                        j = j.item()
                        all_rows.append({
                            "date": date,
                            "realization_id": realization_id,
                            "se": coords[i, j, 0].item(),
                            "sn": coords[i, j, 1].item(),
                            "bearing": bearing[i, j, 0].item(),
                            "length": length[i, j, 0].item(),
                            "width": width[i, j, 0].item(),
                            "ef": ef_class[i, j].item(),
                            "exists_prob": exists_probs[j].item(),
                        })

        gen_counts_per_day.append(day_gen_counts)

    # Build DataFrames
    df = pd.DataFrame(all_rows)
    df.to_parquet(os.path.join(output_dir, "samples.parquet"), index=False)

    gt_df = pd.DataFrame(gt_rows)
    gt_df.to_parquet(os.path.join(output_dir, "gt_tracks.parquet"), index=False)

    env_df = pd.DataFrame(env_rows)
    env_df.to_parquet(os.path.join(output_dir, "environment.parquet"), index=False)

    exists_df = pd.DataFrame(exists_rows)
    exists_df.to_parquet(os.path.join(output_dir, "exists_probs.parquet"), index=False)

    # Compute metrics
    summary = _compute_metrics(gt_counts, gen_counts_per_day, df)
    summary["n_days"] = len(ds)

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Plots (optional — skip if matplotlib/cartopy not available)
    try:
        _save_plots(gt_counts, gen_counts_per_day, df, output_dir)
    except ImportError:
        pass

    return summary


def _compute_metrics(
    gt_counts: list[int],
    gen_counts_per_day: list[list[int]],
    df: pd.DataFrame,
) -> dict[str, Any]:
    """Compute evaluation metrics."""
    gt_arr = np.array(gt_counts)

    # Mean generated count per day (average over realizations)
    mean_gen_counts = np.array([np.mean(c) for c in gen_counts_per_day])

    # Count MAE
    count_mae = float(np.abs(mean_gen_counts - gt_arr).mean())

    # KS test on count distributions
    all_gen_counts = [c for day_counts in gen_counts_per_day for c in day_counts]
    # Compare generated count distribution to GT count distribution
    # (repeat GT counts to match number of realizations)
    n_samples_per_day = len(gen_counts_per_day[0]) if gen_counts_per_day else 1
    gt_repeated = np.repeat(gt_arr, n_samples_per_day)
    if len(all_gen_counts) > 0 and len(gt_repeated) > 0:
        ks_stat, ks_pvalue = stats.ks_2samp(all_gen_counts, gt_repeated)
    else:
        ks_stat, ks_pvalue = 0.0, 1.0

    # Per-day count std (sample diversity)
    count_stds = [float(np.std(c)) for c in gen_counts_per_day]
    mean_count_std = float(np.mean(count_stds)) if count_stds else 0.0

    # Null day rate
    gt_null_rate = float((gt_arr == 0).mean()) if len(gt_arr) > 0 else 0.0
    gen_null_rate = float(np.mean([
        np.mean([c == 0 for c in day_counts])
        for day_counts in gen_counts_per_day
    ])) if gen_counts_per_day else 0.0

    result: dict[str, Any] = {
        "count_mae": count_mae,
        "count_ks_stat": float(ks_stat),
        "count_ks_pvalue": float(ks_pvalue),
        "mean_count_std": mean_count_std,
        "gt_null_rate": gt_null_rate,
        "gen_null_rate": gen_null_rate,
    }

    # Track property KS tests (if we have generated tracks)
    if len(df) > 0:
        # Gather all GT tracks for comparison
        # (we don't have direct access to GT tracks here, so just report gen stats)
        for col in ["bearing", "length", "width"]:
            if col in df.columns:
                result[f"gen_{col}_mean"] = float(df[col].mean())
                result[f"gen_{col}_std"] = float(df[col].std())

        # EF distribution
        if "ef" in df.columns:
            ef_counts = df["ef"].value_counts().sort_index()
            result["gen_ef_distribution"] = {int(k): int(v) for k, v in ef_counts.items()}

    return result


def _save_plots(
    gt_counts: list[int],
    gen_counts_per_day: list[list[int]],
    df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Save diagnostic plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. Count distribution comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    all_gen = [c for day_counts in gen_counts_per_day for c in day_counts]
    max_count = max(max(gt_counts, default=0), max(all_gen, default=0), 1)
    bins = np.arange(-0.5, max_count + 1.5, 1)
    ax.hist(gt_counts, bins=bins, alpha=0.6, label="Ground Truth", density=True)
    ax.hist(all_gen, bins=bins, alpha=0.6, label="Generated", density=True)
    ax.set_xlabel("Track Count")
    ax.set_ylabel("Density")
    ax.set_title("Track Count Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "count_distribution.png"), dpi=150)
    plt.close(fig)

    # 2. Per-day count comparison
    mean_gen = [np.mean(c) for c in gen_counts_per_day]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(gt_counts, mean_gen, alpha=0.6, s=20)
    lim = max(max(gt_counts, default=1), max(mean_gen, default=1)) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3)
    ax.set_xlabel("GT Count")
    ax.set_ylabel("Mean Generated Count")
    ax.set_title("Per-Day Count: GT vs Generated")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "count_scatter.png"), dpi=150)
    plt.close(fig)

    # 3. Genesis density (if we have tracks)
    if len(df) > 0 and "se" in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, title, data_se, data_sn in []:
            pass  # placeholder for GT comparison (needs GT track access)
        # Just plot generated genesis density
        ax = axes[0]
        ax.hist2d(df["se"].values, df["sn"].values, bins=20,
                  range=[[0, 1], [0, 1]], cmap="YlOrRd")
        ax.set_title("Generated Genesis Density")
        ax.set_xlabel("easting")
        ax.set_ylabel("northing")
        ax.set_aspect("equal")
        axes[1].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "genesis_density.png"), dpi=150)
        plt.close(fig)
