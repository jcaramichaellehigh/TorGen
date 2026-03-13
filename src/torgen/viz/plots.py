"""Visual sanity checks for TorGen model output."""
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

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

    Args:
        model: Trained TorGenCVAE (will be set to eval mode).
        pt_path: Path to a single .pt file (e.g., 2011-04-27.pt).
        n_samples: Number of z samples to generate.
        threshold: Exists probability threshold for predicted tracks.
        figsize: Size of each subplot panel.
        save_path: If provided, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    from torgen.data.dataset import _load_pt
    sample = _load_pt(pt_path)
    wx = sample["wx"].unsqueeze(0)  # (1, C, H, W)
    gt_tracks = sample["tracks"]     # (N, 6)
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
        exists = preds["exists"][0].squeeze(-1)  # (Q,)
        coords = preds["coords"][0]               # (Q, 4)
        mask = exists > threshold
        pred_tracks = coords[mask]                 # (K, 4)

        # Build (K, 6) with dummy width/ef for plotting
        if pred_tracks.shape[0] > 0:
            tracks_for_plot = torch.zeros(pred_tracks.shape[0], 6)
            tracks_for_plot[:, :4] = pred_tracks.cpu()
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
        tracks: (N, 6) tensor — columns: se, sn, ee, en, width, ef.
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

    tracks_np = tracks.detach().cpu().numpy()
    for t in tracks_np:
        se, sn, ee, en = t[0], t[1], t[2], t[3]
        ef = int(t[5]) if t.shape[0] > 5 else 0
        color = _ef_color(ef)
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
