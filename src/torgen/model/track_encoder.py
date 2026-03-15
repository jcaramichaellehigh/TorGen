"""Posterior Track Set Encoder.

Summarises ground-truth storm tracks into a fixed-size vector for the VAE
posterior.  Uses transformer layers with self-attention (inter-track
relationships) and cross-attention (contextualising tracks against the
spatial weather map from the CNN encoder).

Null days (0 tracks) short-circuit to a learned null embedding so the
network never has to run attention on an empty sequence.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PosteriorTrackEncoderLayer(nn.Module):
    """One layer: self-attention among tracks, cross-attention to weather, FFN."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(
        self,
        track_tokens: torch.Tensor,
        spatial_tokens: torch.Tensor,
        track_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Self-attention among track tokens
        h = self.norm1(track_tokens)
        h2, _ = self.self_attn(h, h, h, key_padding_mask=track_key_padding_mask)
        track_tokens = track_tokens + self.drop1(h2)

        # Cross-attention to spatial weather tokens
        h = self.norm2(track_tokens)
        h2, _ = self.cross_attn(h, spatial_tokens, spatial_tokens)
        track_tokens = track_tokens + self.drop2(h2)

        # FFN
        h = self.norm3(track_tokens)
        track_tokens = track_tokens + self.drop3(self.ffn(h))
        return track_tokens


class PosteriorTrackEncoder(nn.Module):
    """Encodes ground truth track set into a summary vector for the posterior.

    Handles null days (0 tracks) by returning a learned null embedding.
    """

    def __init__(
        self,
        track_dim: int = 6,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(track_dim, d_model)
        self.layers = nn.ModuleList(
            [PosteriorTrackEncoderLayer(d_model, n_heads, dropout=dropout)
             for _ in range(n_layers)]
        )
        self.null_embedding = nn.Parameter(torch.randn(d_model))

    def forward(
        self,
        tracks: torch.Tensor,
        track_mask: torch.Tensor,
        spatial_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a padded set of storm tracks into a single summary vector.

        Args:
            tracks: (B, N, track_dim) padded track vectors.
            track_mask: (B, N) boolean -- True for real tracks.
            spatial_tokens: (B, S, d_model) flattened spatial feature map.

        Returns:
            summary: (B, d_model)
        """
        B = tracks.shape[0]

        # Handle empty track sets -- nothing to attend over.
        if tracks.shape[1] == 0:
            return self.null_embedding.unsqueeze(0).expand(B, -1)

        tokens = self.embed(tracks)  # (B, N, d_model)

        # MHA key_padding_mask: True = ignore, so invert our mask.
        padding_mask = ~track_mask

        # For null-day samples (all tracks masked), unmask everything to avoid
        # NaN from softmax over a fully-masked sequence. The output for these
        # samples is replaced by null_embedding below, so attention values
        # don't matter — we just need gradient-safe computation.
        no_tracks = ~track_mask.any(dim=1)  # (B,)
        if no_tracks.any():
            padding_mask = padding_mask.clone()
            padding_mask[no_tracks] = False

        for layer in self.layers:
            tokens = layer(tokens, spatial_tokens, track_key_padding_mask=padding_mask)

        # Mean-pool over real tracks only.
        mask_expanded = track_mask.unsqueeze(-1).float()  # (B, N, 1)
        counts = mask_expanded.sum(dim=1).clamp(min=1)  # (B, 1)
        summary = (tokens * mask_expanded).sum(dim=1) / counts

        # For any samples that had no real tracks, substitute null embedding.
        no_tracks = ~track_mask.any(dim=1)  # (B,)
        if no_tracks.any():
            summary[no_tracks] = self.null_embedding

        return summary
