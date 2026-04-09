# src/torgen/loss/sequence.py
"""Per-position sequence loss for autoregressive track generation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _circular_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 loss on circular [0, 1] domain: min(|d|, 1 - |d|)."""
    diff = (pred - target).abs()
    return torch.min(diff, 1.0 - diff)


class SequenceLoss(nn.Module):
    """Per-position loss for autoregressive track sequence.

    For each position i in the sequence:
    - Positions 0..n-1: predict track i+1 (regression + classification + stop=0)
    - Position n: predict STOP (stop=1, no regression losses)
    """

    def __init__(
        self,
        lambda_coord: float = 5.0,
        lambda_bearing: float = 2.0,
        lambda_length: float = 2.0,
        lambda_width: float = 2.0,
        lambda_ef: float = 2.0,
        lambda_stop: float = 1.0,
        n_ef_classes: int = 6,
        ef_class_weights: torch.Tensor | None = None,
        ef_weight_power: float = 0.0,
    ) -> None:
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_bearing = lambda_bearing
        self.lambda_length = lambda_length
        self.lambda_width = lambda_width
        self.lambda_ef = lambda_ef
        self.lambda_stop = lambda_stop
        self.n_ef_classes = n_ef_classes
        self.ef_weight_power = ef_weight_power
        if ef_class_weights is not None:
            self.register_buffer("ef_class_weights", ef_class_weights)
        else:
            self.ef_class_weights = None

    def forward(self, preds: dict[str, torch.Tensor],
                tracks: torch.Tensor,
                track_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute sequence loss.

        Args:
            preds: dict from TrackSequenceDecoder.forward() with shape (B, seq_len, ...).
                   seq_len = max_N + 1 (START position predicts first track, etc.)
            tracks: (B, max_N, 6) GT tracks, zero-padded.
            track_mask: (B, max_N) bool mask for real tracks.

        Returns:
            Loss dict with keys: total, coord, bearing, length, width, ef, stop.
        """
        B = tracks.shape[0]
        max_n = tracks.shape[1]
        device = preds["coords"].device

        pred_coords = preds["coords"]       # (B, seq_len, 2)
        pred_bearing = preds["bearing"]      # (B, seq_len, 1)
        pred_length = preds["length"]        # (B, seq_len, 1)
        pred_width = preds["width"]          # (B, seq_len, 1)
        pred_ef = preds["ef_logits"]         # (B, seq_len, 6)
        pred_stop = preds["stop"]            # (B, seq_len, 1)

        n_tracks = track_mask.sum(dim=1)  # (B,)
        seq_len = max_n + 1

        # Stop targets and mask
        stop_targets = torch.zeros(B, seq_len, device=device)
        stop_mask = torch.zeros(B, seq_len, dtype=torch.bool, device=device)

        for b in range(B):
            n = n_tracks[b].item()
            stop_mask[b, :n] = True      # track positions are real
            stop_mask[b, n] = True        # stop position is real
            stop_targets[b, n] = 1.0      # stop here

        # Regression losses (over track positions only)
        total_coord = torch.tensor(0.0, device=device)
        total_bearing = torch.tensor(0.0, device=device)
        total_length = torch.tensor(0.0, device=device)
        total_width = torch.tensor(0.0, device=device)
        total_ef = torch.tensor(0.0, device=device)
        n_total_tracks = 0

        for b in range(B):
            n = n_tracks[b].item()
            if n == 0:
                continue

            p_coords = pred_coords[b, :n]
            p_bearing = pred_bearing[b, :n]
            p_length = pred_length[b, :n]
            p_width = pred_width[b, :n]
            p_ef = pred_ef[b, :n]

            gt = tracks[b, :n]

            total_coord = total_coord + F.l1_loss(
                p_coords, gt[:, :2], reduction="sum")
            total_bearing = total_bearing + _circular_l1(
                p_bearing, gt[:, 2:3]).sum()
            total_length = total_length + F.l1_loss(
                p_length, gt[:, 3:4], reduction="sum")
            total_width = total_width + F.smooth_l1_loss(
                p_width, gt[:, 4:5], reduction="sum")
            total_ef = total_ef + F.cross_entropy(
                p_ef, gt[:, 5].long(),
                weight=self.ef_class_weights, reduction="sum")
            n_total_tracks += n

        n_total_tracks = max(n_total_tracks, 1)

        # Stop loss (over all real positions: tracks + stop)
        real_stop_preds = pred_stop.squeeze(-1)[stop_mask]
        real_stop_targets = stop_targets[stop_mask]
        real_stop_preds = real_stop_preds.clamp(1e-6, 1 - 1e-6)
        total_stop = F.binary_cross_entropy(
            real_stop_preds, real_stop_targets, reduction="sum")
        n_total_positions = max(stop_mask.sum().item(), 1)

        # Weighted losses
        coord_loss = self.lambda_coord * total_coord / n_total_tracks
        bearing_loss = self.lambda_bearing * total_bearing / n_total_tracks
        length_loss = self.lambda_length * total_length / n_total_tracks
        width_loss = self.lambda_width * total_width / n_total_tracks
        ef_loss = self.lambda_ef * total_ef / n_total_tracks
        stop_loss = self.lambda_stop * total_stop / n_total_positions

        total = (coord_loss + bearing_loss + length_loss + width_loss
                 + ef_loss + stop_loss)

        return {
            "total": total,
            "coord": coord_loss,
            "bearing": bearing_loss,
            "length": length_loss,
            "width": width_loss,
            "ef": ef_loss,
            "stop": stop_loss,
        }
