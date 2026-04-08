"""Permutation-invariant loss with Hungarian (optimal) matching (v2)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def _focal_bce(
    pred: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    reduction: str = "sum",
) -> torch.Tensor:
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    p_t = pred * target + (1 - pred) * (1 - target)
    focal_weight = (1 - p_t) ** gamma
    loss = focal_weight * bce
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


def _circular_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 loss on circular [0, 1] domain: min(|d|, 1 - |d|)."""
    diff = (pred - target).abs()
    return torch.min(diff, 1.0 - diff)


class HungarianMatchingLoss(nn.Module):

    def __init__(
        self,
        lambda_coord: float = 5.0,
        lambda_bearing: float = 2.0,
        lambda_length: float = 2.0,
        lambda_width: float = 2.0,
        lambda_ef: float = 2.0,
        lambda_exists: float = 2.0,
        lambda_noobj: float = 2.0,
        focal_gamma: float = 2.0,
        focal_gamma_noobj: float = 2.0,
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
        self.lambda_exists = lambda_exists
        self.lambda_noobj = lambda_noobj
        self.focal_gamma = focal_gamma
        self.focal_gamma_noobj = focal_gamma_noobj
        self.n_ef_classes = n_ef_classes
        self.ef_weight_power = ef_weight_power
        if ef_class_weights is not None:
            self.register_buffer("ef_class_weights", ef_class_weights)
        else:
            self.ef_class_weights = None

    @torch.no_grad()
    def _match(self, pred_exists, pred_coords, pred_bearing, pred_length,
               pred_width, pred_ef_logits, gt_tracks):
        N = gt_tracks.shape[0]
        if N == 0:
            return [], []

        gt_coords = gt_tracks[:, :2]
        gt_bearing = gt_tracks[:, 2:3]
        gt_length = gt_tracks[:, 3:4]
        gt_width = gt_tracks[:, 4:5]
        gt_ef = gt_tracks[:, 5].long()

        cost_coord = torch.cdist(pred_coords, gt_coords, p=1)

        # Circular distance for bearing (values in [0, 1])
        bearing_diff = (pred_bearing - gt_bearing.T).abs()
        cost_bearing = torch.min(bearing_diff, 1.0 - bearing_diff)

        cost_length = torch.cdist(pred_length, gt_length, p=1)
        cost_width = torch.cdist(pred_width, gt_width, p=1)

        log_probs = F.log_softmax(pred_ef_logits, dim=-1)
        cost_ef = -log_probs[:, gt_ef]

        cost_exists = (
            -torch.log(pred_exists.squeeze(-1) + 1e-8).unsqueeze(1).expand(-1, N)
        )

        cost = (
            self.lambda_coord * cost_coord
            + self.lambda_bearing * cost_bearing
            + self.lambda_length * cost_length
            + self.lambda_width * cost_width
            + self.lambda_ef * cost_ef
            + self.lambda_exists * cost_exists
        )

        pred_idx, gt_idx = linear_sum_assignment(cost.cpu().numpy())
        return list(pred_idx), list(gt_idx)

    def forward(self, preds, tracks, track_mask):
        B = tracks.shape[0]
        device = preds["exists"].device

        total_coord = torch.tensor(0.0, device=device)
        total_bearing = torch.tensor(0.0, device=device)
        total_length = torch.tensor(0.0, device=device)
        total_width = torch.tensor(0.0, device=device)
        total_ef = torch.tensor(0.0, device=device)
        total_exists = torch.tensor(0.0, device=device)
        total_noobj = torch.tensor(0.0, device=device)
        n_matched = 0
        total_Q = 0

        for b in range(B):
            n_gt = track_mask[b].sum().item()
            gt = tracks[b, :n_gt]

            pred_exists = preds["exists"][b].clamp(1e-6, 1 - 1e-6)
            pred_coords = preds["coords"][b]
            pred_bearing = preds["bearing"][b]
            pred_length = preds["length"][b]
            pred_width = preds["width"][b]
            pred_ef_logits = preds["ef_logits"][b]

            pred_idx, gt_idx = self._match(
                pred_exists, pred_coords, pred_bearing, pred_length,
                pred_width, pred_ef_logits, gt,
            )

            Q = pred_exists.shape[0]
            total_Q += Q
            matched_set = set(pred_idx)
            unmatched_idx = [i for i in range(Q) if i not in matched_set]

            if len(pred_idx) > 0:
                pi = torch.tensor(pred_idx, device=device)
                gi = torch.tensor(gt_idx, device=device)

                total_coord = total_coord + F.l1_loss(
                    pred_coords[pi], gt[:, :2][gi], reduction="sum")
                total_bearing = total_bearing + _circular_l1(
                    pred_bearing[pi], gt[:, 2:3][gi]).sum()
                total_length = total_length + F.l1_loss(
                    pred_length[pi], gt[:, 3:4][gi], reduction="sum")
                total_width = total_width + F.smooth_l1_loss(
                    pred_width[pi], gt[:, 4:5][gi], reduction="sum")
                gt_ef_classes = gt[:, 5][gi].long()
                total_ef = total_ef + F.cross_entropy(
                    pred_ef_logits[pi], gt_ef_classes,
                    weight=self.ef_class_weights, reduction="sum")
                total_exists = total_exists + _focal_bce(
                    pred_exists[pi].squeeze(-1),
                    torch.ones(len(pred_idx), device=device),
                    gamma=self.focal_gamma)
                n_matched += len(pred_idx)

            if len(unmatched_idx) > 0:
                ui = torch.tensor(unmatched_idx, device=device)
                total_noobj = total_noobj + _focal_bce(
                    pred_exists[ui].squeeze(-1),
                    torch.zeros(len(unmatched_idx), device=device),
                    gamma=self.focal_gamma_noobj)

        n_matched = max(n_matched, 1)
        total_Q = max(total_Q, 1)

        coord_loss = self.lambda_coord * total_coord / n_matched
        bearing_loss = self.lambda_bearing * total_bearing / n_matched
        length_loss = self.lambda_length * total_length / n_matched
        width_loss = self.lambda_width * total_width / n_matched
        ef_loss = self.lambda_ef * total_ef / n_matched
        exists_loss = self.lambda_exists * total_exists / n_matched
        noobj_loss = self.lambda_noobj * total_noobj / total_Q

        total = (coord_loss + bearing_loss + length_loss + width_loss
                 + ef_loss + exists_loss + noobj_loss)

        return {
            "total": total,
            "coord": coord_loss,
            "bearing": bearing_loss,
            "length": length_loss,
            "width": width_loss,
            "ef": ef_loss,
            "exists": exists_loss,
            "noobj": noobj_loss,
        }
