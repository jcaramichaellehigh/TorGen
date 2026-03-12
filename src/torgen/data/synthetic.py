import os
import torch


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
        tracks: (N, 6) float32 -- N can be 0 for null days
        date: str YYYY-MM-DD
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
            coords = torch.rand(n_tracks, 4, generator=rng)  # se, sn, ee, en
            width = torch.rand(n_tracks, 1, generator=rng)
            ef = torch.randint(0, 6, (n_tracks, 1), generator=rng).float()
            tracks = torch.cat([coords, width, ef], dim=1)

        date = f"2000-{(i // 28) + 3:02d}-{(i % 28) + 1:02d}"

        torch.save(
            {"wx": wx, "tracks": tracks, "date": date},
            os.path.join(output_dir, f"{date}.pt"),
        )
