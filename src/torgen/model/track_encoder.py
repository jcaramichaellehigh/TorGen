"""Posterior Track Set Encoder (v2).

Linear embedding + mean pool. No transformer layers.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PosteriorTrackEncoder(nn.Module):
    """Encodes ground truth track set into a summary vector for the posterior.

    Linear embedding per track, mean pool over real tracks.
    Null days return a learned null embedding.
    """

    def __init__(self, track_dim: int = 6, d_model: int = 256) -> None:
        super().__init__()
        self.embed = nn.Linear(track_dim, d_model)
        self.null_embedding = nn.Parameter(torch.randn(d_model))

    def forward(self, tracks: torch.Tensor,
                track_mask: torch.Tensor) -> torch.Tensor:
        B = tracks.shape[0]

        if tracks.shape[1] == 0:
            return self.null_embedding.unsqueeze(0).expand(B, -1)

        tokens = self.embed(tracks)

        mask_expanded = track_mask.unsqueeze(-1).float()
        counts = mask_expanded.sum(dim=1).clamp(min=1)
        summary = (tokens * mask_expanded).sum(dim=1) / counts

        no_tracks = ~track_mask.any(dim=1)
        if no_tracks.any():
            summary[no_tracks] = self.null_embedding

        return summary
