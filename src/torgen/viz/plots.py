"""Visual sanity checks for TorGen model output."""
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from torgen.data.dataset import bearing_length_to_coords, coords_to_bearing_length
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
    from torgen.data.dataset import _load_pt
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
