"""Visual sanity checks for TorGen model output."""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from torgen.data.dataset import bearing_length_to_coords, coords_to_bearing_length, _load_pt
from torgen.model.cvae import TorGenCVAE


def plot_outbreak_comparison(
    model: TorGenCVAE,
    pt_path: str,
    n_samples: int = 5,
    threshold: float = 0.5,
    figsize: tuple[float, float] = (4.0, 4.0),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot ground truth tracks vs. sampled realizations in [0,1] space.

    Produces a (1 + n_samples) panel figure: ground truth on the left,
    then n_samples independent generations from the same weather.
    """
    sample = _load_pt(pt_path)
    wx = sample["wx"].unsqueeze(0)  # (1, C, H, W)
    gt_tracks = coords_to_bearing_length(sample["tracks"])  # (N, 6)
    date = sample.get("date", Path(pt_path).stem)

    device = next(model.parameters()).device
    wx = wx.to(device)

    model.eval()

    n_cols = 1 + n_samples
    fig, axes = plt.subplots(1, n_cols, figsize=(figsize[0] * n_cols, figsize[1]))
    if n_cols == 1:
        axes = [axes]

    # Ground truth panel
    _plot_tracks(axes[0], gt_tracks, title=f"Ground Truth\n{date}")

    # Sampled panels
    for i in range(n_samples):
        with torch.no_grad():
            out = model.generate(wx)
        preds = out["preds"]
        exists = preds["exists"][0].squeeze(-1)        # (Q,)
        coords = preds["coords"][0]                     # (Q, 2)
        bearing = preds["bearing"][0]                    # (Q, 1)
        length = preds["length"][0]                      # (Q, 1)
        width = preds["width"][0]                        # (Q, 1)
        mask = exists > threshold

        if mask.any():
            # Reassemble (K, 6) in bearing/length format for plotting
            tracks_for_plot = torch.zeros(mask.sum().item(), 6)
            tracks_for_plot[:, :2] = coords[mask].cpu()
            tracks_for_plot[:, 2] = bearing[mask].squeeze(-1).cpu()
            tracks_for_plot[:, 3] = length[mask].squeeze(-1).cpu()
            tracks_for_plot[:, 4] = width[mask].squeeze(-1).cpu()
            # ef defaults to 0
        else:
            tracks_for_plot = torch.zeros(0, 6)

        n_tracks = tracks_for_plot.shape[0]
        _plot_tracks(axes[1 + i], tracks_for_plot, title=f"Sample {i + 1}\n({n_tracks} tracks)")

    fig.suptitle(f"TorGen: {date}", fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def _plot_tracks(ax: plt.Axes, tracks: torch.Tensor, title: str = "") -> None:
    """Plot tornado tracks as lines in [0,1] x [0,1] space.

    Args:
        ax: Matplotlib axes.
        tracks: (N, 6) tensor -- columns: se, sn, bearing_norm, length_norm, width, ef.
            Converted back to (se, sn, ee, en) for plotting.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("easting")
    ax.set_ylabel("northing")

    if tracks.shape[0] == 0:
        ax.text(0.5, 0.5, "No tracks", ha="center", va="center",
                fontsize=10, color="gray", transform=ax.transAxes)
        return

    coords = bearing_length_to_coords(tracks).detach().cpu().numpy()
    ef_vals = tracks[:, 5].detach().cpu().numpy() if tracks.shape[1] > 5 else [0] * tracks.shape[0]
    for i in range(len(coords)):
        se, sn, ee, en = coords[i, 0], coords[i, 1], coords[i, 2], coords[i, 3]
        color = _ef_color(int(ef_vals[i]))
        ax.plot([se, ee], [sn, en], color=color, linewidth=1.0, alpha=0.8)


def _ef_color(ef: int) -> str:
    """Map EF rating to color."""
    colors = {
        0: "#2196F3",  # blue
        1: "#4CAF50",  # green
        2: "#FFC107",  # amber
        3: "#FF9800",  # orange
        4: "#F44336",  # red
        5: "#9C27B0",  # purple
    }
    return colors.get(ef, "#757575")


def plot_training_curves(loss_history: dict[str, list[float]],
                         save_path: str | None = None) -> plt.Figure:
    """Plot decomposed training curves from loss_history.

    Produces a 2x2 figure:
    - Top-left: total train/val loss
    - Top-right: KL vs reconstruction (train), beta on secondary axis
    - Bottom-left: per-component reconstruction (train)
    - Bottom-right: per-component reconstruction (val)

    Handles lists of different lengths -- only plots series with len > 0.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    def _epochs(series: list[float]) -> range:
        return range(1, len(series) + 1)

    # Top-left: total train/val
    ax = axes[0, 0]
    if len(loss_history.get("train_total", [])) > 0:
        ax.plot(_epochs(loss_history["train_total"]),
                loss_history["train_total"], label="Train")
    if len(loss_history.get("val_total", [])) > 0:
        ax.plot(_epochs(loss_history["val_total"]),
                loss_history["val_total"], label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Total Loss")
    ax.legend()

    # Top-right: KL vs recon (train) + beta overlay
    ax = axes[0, 1]
    has_kl = len(loss_history.get("train_kl", [])) > 0
    has_recon = len(loss_history.get("train_recon", [])) > 0
    if has_kl:
        ax.plot(_epochs(loss_history["train_kl"]),
                loss_history["train_kl"], label="KL", color="tab:red")
    if has_recon:
        ax.plot(_epochs(loss_history["train_recon"]),
                loss_history["train_recon"], label="Recon", color="tab:blue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("KL vs Reconstruction (train)")
    if has_kl or has_recon:
        ax.legend(loc="upper left")
    if len(loss_history.get("beta", [])) > 0:
        ax2 = ax.twinx()
        ax2.plot(_epochs(loss_history["beta"]),
                 loss_history["beta"], "--", color="gray", alpha=0.6, label="beta")
        ax2.set_ylabel("beta")
        ax2.set_ylim(-0.05, 1.15)
        ax2.legend(loc="upper right")

    # Bottom-left: per-component (train)
    ax = axes[1, 0]
    components = ["coord", "bearing", "length", "width", "ef", "stop"]
    for comp in components:
        key = f"train_{comp}"
        if len(loss_history.get(key, [])) > 0:
            ax.plot(_epochs(loss_history[key]), loss_history[key], label=comp)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Component Losses (train)")
    ax.legend()

    # Bottom-right: per-component (val)
    ax = axes[1, 1]
    for comp in components:
        key = f"val_{comp}"
        if len(loss_history.get(key, [])) > 0:
            ax.plot(_epochs(loss_history[key]), loss_history[key], label=comp)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Component Losses (val)")
    ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_diversity_grid(model: nn.Module,
                        data_dir: str,
                        days: list[str],
                        n_samples: int = 8,
                        stop_threshold: float = 0.5,
                        save_path: str | None = None) -> plt.Figure:
    """Plot multiple z-draw realizations for selected days.

    Rows = days, columns = [GT, sample_1, ..., sample_n].
    Each cell shows tornado tracks in [0,1] space using _plot_tracks.
    """
    import os

    device = next(model.parameters()).device
    n_rows = len(days)
    n_cols = 1 + n_samples
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]  # ensure 2D

    model.eval()

    for row, day_file in enumerate(days):
        path = os.path.join(data_dir, day_file)
        sample = _load_pt(path)
        gt_tracks = coords_to_bearing_length(sample["tracks"])
        date = sample.get("date", day_file.replace(".pt.gz", "").replace(".pt", ""))
        n_gt = gt_tracks.shape[0]

        _plot_tracks(axes[row, 0], gt_tracks,
                     title=f"GT — {date}\n({n_gt} tracks)")

        wx = sample["wx"].unsqueeze(0).to(device)

        for col in range(n_samples):
            with torch.no_grad():
                out = model.generate(wx, stop_threshold=stop_threshold)

            preds = out["preds"]
            gen_mask = preds["gen_mask"][0]

            if gen_mask.any():
                coords = preds["coords"][0][gen_mask]
                bearing = preds["bearing"][0][gen_mask]
                length = preds["length"][0][gen_mask]
                width = preds["width"][0][gen_mask]
                ef = preds["ef_logits"][0][gen_mask].argmax(dim=-1).float()

                tracks_for_plot = torch.zeros(coords.shape[0], 6)
                tracks_for_plot[:, :2] = coords.cpu()
                tracks_for_plot[:, 2] = bearing.squeeze(-1).cpu()
                tracks_for_plot[:, 3] = length.squeeze(-1).cpu()
                tracks_for_plot[:, 4] = width.squeeze(-1).cpu()
                tracks_for_plot[:, 5] = ef.cpu()
            else:
                tracks_for_plot = torch.zeros(0, 6)

            n_pred = tracks_for_plot.shape[0]
            _plot_tracks(axes[row, 1 + col], tracks_for_plot,
                         title=f"Sample {col + 1}\n({n_pred} tracks)")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
