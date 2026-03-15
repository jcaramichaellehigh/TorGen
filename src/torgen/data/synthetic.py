import math
import os

import torch


def _make_synthetic_tracks(
    n_tracks: int, rng: torch.Generator,
) -> torch.Tensor:
    """Generate synthetic tracks in raw (se, sn, ee, en, width, ef) format.

    Tracks have NE-ish bearings and short lengths to mimic real tornadoes.
    The dataset __getitem__ converts to bearing/length on the fly.
    """
    se = torch.rand(n_tracks, 1, generator=rng)
    sn = torch.rand(n_tracks, 1, generator=rng)
    # NE bearing cluster: ~30-60 degrees in compass bearing (0=north, CW)
    bearing = torch.rand(n_tracks, 1, generator=rng) * math.radians(30) + math.radians(30)
    # Short lengths: 0.01 to 0.15 in [0,1] grid space
    length = torch.rand(n_tracks, 1, generator=rng) * 0.14 + 0.01
    ee = se + length * torch.sin(bearing)
    en = sn + length * torch.cos(bearing)
    width = torch.rand(n_tracks, 1, generator=rng)
    ef = torch.randint(0, 6, (n_tracks, 1), generator=rng).float()
    return torch.cat([se, sn, ee, en, width, ef], dim=1)


def generate_synthetic_dataset(
    output_dir: str,
    n_samples: int = 20,
    grid_h: int = 270,
    grid_w: int = 270,
    n_channels: int = 16,
    max_tracks: int = 10,
    seed: int = 42,
) -> None:
    """Generate fake .pt files matching the real data format.

    Each file contains:
        wx: (n_channels, grid_h, grid_w) float32 in [0, 1]
        tracks: (N, 6) float32 -- raw (se, sn, ee, en, width, ef)
        date: str YYYY-MM-DD

    Note: tracks are stored in raw coordinate format. The dataset
    converts to (se, sn, bearing_norm, length_norm, width, ef) on read.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = torch.Generator().manual_seed(seed)

    for i in range(n_samples):
        wx = torch.rand(n_channels, grid_h, grid_w, generator=rng)

        # ~30% chance of null day
        if torch.rand(1, generator=rng).item() < 0.3:
            tracks = torch.zeros(0, 6)
        else:
            n_tracks = int(torch.randint(1, max_tracks + 1, (1,), generator=rng).item())
            tracks = _make_synthetic_tracks(n_tracks, rng)

        date = f"2000-{(i // 28) + 3:02d}-{(i % 28) + 1:02d}"

        torch.save(
            {"wx": wx, "tracks": tracks, "date": date},
            os.path.join(output_dir, f"{date}.pt"),
        )

    # Always include a 2011-04-27 sample (Super Outbreak) with tracks
    wx = torch.rand(n_channels, grid_h, grid_w, generator=rng)
    n_tracks = int(torch.randint(3, max_tracks + 1, (1,), generator=rng).item())
    tracks = _make_synthetic_tracks(n_tracks, rng)
    torch.save(
        {"wx": wx, "tracks": tracks, "date": "2011-04-27"},
        os.path.join(output_dir, "2011-04-27.pt"),
    )
