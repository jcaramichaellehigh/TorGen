"""Permutation-invariant loss with Hungarian (optimal) matching.

Matches predicted track slots to ground truth tracks using a cost matrix,
then computes component losses on matched/unmatched pairs. The matching
itself is non-differentiable (@torch.no_grad), but all downstream losses
flow gradients normally.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class HungarianMatchingLoss(nn.Module):
    """Permutation-invariant loss with Hungarian matching.

    Matches predicted track slots to ground truth tracks using a cost matrix,
    then computes component losses on matched/unmatched pairs.
    """

    def __init__(
        self,
        lambda_coord: float = 5.0,
        lambda_width: float = 2.0,
        lambda_ef: float = 2.0,
        lambda_exists: float = 2.0,
        lambda_noobj: float = 1.0,
        n_ef_classes: int = 6,
        ef_class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_width = lambda_width
        self.lambda_ef = lambda_ef
        self.lambda_exists = lambda_exists
        self.lambda_noobj = lambda_noobj
        self.n_ef_classes = n_ef_classes
        if ef_class_weights is not None:
            self.register_buffer("ef_class_weights", ef_class_weights)
        else:
            self.ef_class_weights = None

    @torch.no_grad()
    def _match(
        self,
        pred_exists: torch.Tensor,
        pred_coords: torch.Tensor,
        pred_width: torch.Tensor,
        pred_ef_logits: torch.Tensor,
        gt_tracks: torch.Tensor,
    ) -> tuple[list[int], list[int]]:
        """Compute cost matrix and return optimal (pred_idx, gt_idx) pairs."""
        N = gt_tracks.shape[0]
        if N == 0:
            return [], []

        gt_coords = gt_tracks[:, :4]
        gt_width = gt_tracks[:, 4:5]
        gt_ef = gt_tracks[:, 5].long()

        # Cost components
        cost_coord = torch.cdist(pred_coords, gt_coords, p=1)  # (Q, N)
        cost_width = torch.cdist(pred_width, gt_width, p=1)  # (Q, N)

        # CE cost: for each (pred, gt) pair
        log_probs = F.log_softmax(pred_ef_logits, dim=-1)  # (Q, n_ef_classes)
        cost_ef = -log_probs[:, gt_ef]  # (Q, N)

        # Exists cost: how far pred_exists is from 1.0
        cost_exists = (
            -torch.log(pred_exists.squeeze(-1) + 1e-8).unsqueeze(1).expand(-1, N)
        )

        cost = (
            self.lambda_coord * cost_coord
            + self.lambda_width * cost_width
            + self.lambda_ef * cost_ef
            + self.lambda_exists * cost_exists
        )

        pred_idx, gt_idx = linear_sum_assignment(cost.cpu().numpy())
        return list(pred_idx), list(gt_idx)

    def forward(
        self,
        preds: dict[str, torch.Tensor],
        tracks: torch.Tensor,
        track_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute Hungarian matching loss over a batch.

        Args:
            preds: dict with keys:
                - exists:    (B, Q, 1) sigmoid probabilities
                - coords:    (B, Q, 4) normalized coordinates
                - width:     (B, Q, 1) normalized width
                - ef_logits: (B, Q, n_ef_classes) raw logits
            tracks: (B, N, 6) padded ground truth
                    columns: [coord0, coord1, coord2, coord3, width, ef_class]
            track_mask: (B, N) bool mask for valid ground truth tracks

        Returns:
            dict with total, coord, width, ef, exists, noobj scalar losses
        """
        B = tracks.shape[0]
        device = preds["exists"].device

        total_coord = torch.tensor(0.0, device=device)
        total_width = torch.tensor(0.0, device=device)
        total_ef = torch.tensor(0.0, device=device)
        total_exists = torch.tensor(0.0, device=device)
        total_noobj = torch.tensor(0.0, device=device)
        n_matched = 0
        n_unmatched = 0

        for b in range(B):
            n_gt = track_mask[b].sum().item()
            gt = tracks[b, :n_gt]  # (n_gt, 6)

            pred_exists = preds["exists"][b].clamp(1e-6, 1 - 1e-6)  # (Q, 1)
            pred_coords = preds["coords"][b]  # (Q, 4)
            pred_width = preds["width"][b]  # (Q, 1)
            pred_ef_logits = preds["ef_logits"][b]  # (Q, n_ef_classes)

            pred_idx, gt_idx = self._match(
                pred_exists, pred_coords, pred_width, pred_ef_logits, gt
            )

            Q = pred_exists.shape[0]
            matched_set = set(pred_idx)
            unmatched_idx = [i for i in range(Q) if i not in matched_set]

            # Matched losses
            if len(pred_idx) > 0:
                pi = torch.tensor(pred_idx, device=device)
                gi = torch.tensor(gt_idx, device=device)

                total_coord = total_coord + F.l1_loss(
                    pred_coords[pi], gt[:, :4][gi], reduction="sum"
                )
                total_width = total_width + F.l1_loss(
                    pred_width[pi], gt[:, 4:5][gi], reduction="sum"
                )
                gt_ef_classes = gt[:, 5][gi].long()
                total_ef = total_ef + F.cross_entropy(
                    pred_ef_logits[pi],
                    gt_ef_classes,
                    weight=self.ef_class_weights,
                    reduction="sum",
                )
                total_exists = total_exists + F.binary_cross_entropy(
                    pred_exists[pi].squeeze(-1),
                    torch.ones(len(pred_idx), device=device),
                    reduction="sum",
                )
                n_matched += len(pred_idx)

            # Unmatched slots: push exists toward 0
            if len(unmatched_idx) > 0:
                ui = torch.tensor(unmatched_idx, device=device)
                total_noobj = total_noobj + F.binary_cross_entropy(
                    pred_exists[ui].squeeze(-1),
                    torch.zeros(len(unmatched_idx), device=device),
                    reduction="sum",
                )
                n_unmatched += len(unmatched_idx)

        # Normalize by counts (avoid div by zero)
        n_matched = max(n_matched, 1)
        n_unmatched = max(n_unmatched, 1)

        coord_loss = self.lambda_coord * total_coord / n_matched
        width_loss = self.lambda_width * total_width / n_matched
        ef_loss = self.lambda_ef * total_ef / n_matched
        exists_loss = self.lambda_exists * total_exists / n_matched
        noobj_loss = self.lambda_noobj * total_noobj / n_unmatched

        total = coord_loss + width_loss + ef_loss + exists_loss + noobj_loss

        return {
            "total": total,
            "coord": coord_loss,
            "width": width_loss,
            "ef": ef_loss,
            "exists": exists_loss,
            "noobj": noobj_loss,
        }
