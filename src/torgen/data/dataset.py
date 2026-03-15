import gzip
import io
import math
import os
from typing import Any

import torch
from torch.utils.data import Dataset

_SQRT2 = math.sqrt(2.0)


def _load_pt(path: str) -> dict[str, Any]:
    """Load a .pt or .pt.gz file."""
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return torch.load(io.BytesIO(f.read()), weights_only=False)
    return torch.load(path, weights_only=False)


def coords_to_bearing_length(tracks: torch.Tensor) -> torch.Tensor:
    """Convert track coords from (se, sn, ee, en, ...) to (se, sn, bearing_norm, length_norm, ...).

    bearing = atan2(dx, dy)  — compass bearing where 0=north, pi/4=NE
    bearing_norm = (bearing + pi) / (2*pi)  — maps [-pi, pi] to [0, 1]
    length_norm = length / sqrt(2)  — max possible length is grid diagonal
    """
    if tracks.shape[0] == 0:
        return tracks
    out = tracks.clone()
    dx = tracks[:, 2] - tracks[:, 0]  # ee - se
    dy = tracks[:, 3] - tracks[:, 1]  # en - sn
    bearing = torch.atan2(dx, dy)
    bearing_norm = (bearing + math.pi) / (2 * math.pi)
    length = torch.sqrt(dx ** 2 + dy ** 2)
    length_norm = length / _SQRT2
    out[:, 2] = bearing_norm
    out[:, 3] = length_norm
    return out


def bearing_length_to_coords(tracks: torch.Tensor) -> torch.Tensor:
    """Convert track coords from (se, sn, bearing_norm, length_norm, ...) to (se, sn, ee, en, ...).

    Inverse of coords_to_bearing_length.
    """
    if tracks.shape[0] == 0:
        return tracks
    out = tracks.clone()
    bearing = tracks[:, 2] * (2 * math.pi) - math.pi
    length = tracks[:, 3] * _SQRT2
    out[:, 2] = tracks[:, 0] + length * torch.sin(bearing)  # ee = se + dx
    out[:, 3] = tracks[:, 1] + length * torch.cos(bearing)  # en = sn + dy
    return out


SPLIT_YEARS = {
    "train": range(1996, 2019),
    "val": range(2019, 2022),
    "test": range(2022, 2025),
}


class TornadoDataset(Dataset):
    """Dataset of pre-processed .pt tornado day samples."""

    def __init__(
        self, data_dir: str, preload: bool = False, split: str | None = None,
    ) -> None:
        self.data_dir = data_dir
        all_files = sorted(
            f for f in os.listdir(data_dir)
            if f.endswith(".pt") or f.endswith(".pt.gz")
        )
        if split is not None:
            years = SPLIT_YEARS[split]
            all_files = [f for f in all_files if _parse_year(f) in years]
        self.files = all_files
        self.preload = preload
        self._cache: list[dict[str, Any] | None] = [None] * len(self.files)
        if preload:
            for i in range(len(self.files)):
                self._cache[i] = _load_pt(os.path.join(data_dir, self.files[i]))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._cache[idx] is not None:
            sample = self._cache[idx]
        else:
            sample = _load_pt(os.path.join(self.data_dir, self.files[idx]))
        return {
            **sample,
            "tracks": coords_to_bearing_length(sample["tracks"]),
        }


def tornado_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate variable-length track sets into a padded batch.

    Returns:
        wx: (B, C, H, W)
        tracks: (B, max_N, 6) -- zero-padded
        track_mask: (B, max_N) -- True for real tracks, False for padding
        dates: list[str]
    """
    wx = torch.stack([s["wx"] for s in samples])
    dates = [s["date"] for s in samples]
    track_counts = [s["tracks"].shape[0] for s in samples]
    max_n = max(track_counts) if track_counts else 0

    B = len(samples)
    tracks = torch.zeros(B, max_n, 6)
    track_mask = torch.zeros(B, max_n, dtype=torch.bool)

    for i, s in enumerate(samples):
        n = s["tracks"].shape[0]
        if n > 0:
            tracks[i, :n] = s["tracks"]
            track_mask[i, :n] = True

    return {"wx": wx, "tracks": tracks, "track_mask": track_mask, "dates": dates}


def _parse_year(filename: str) -> int:
    """Extract year from filenames like '2011-04-27.pt'."""
    return int(filename[:4])
