# src/torgen/loss/mdn.py
"""Poisson process NLL loss for MDN-CVAE."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_LOG2PI = math.log(2 * math.pi)


def _gaussian_log_prob(x: torch.Tensor, mu: torch.Tensor,
                       L: torch.Tensor) -> torch.Tensor:
    """Log probability of x under Gaussian(mu, L @ L^T).

    Args:
        x: (B, N, 2) observed locations.
        mu: (B, K, 2) component means.
        L: (B, K, 2, 2) lower-triangular Cholesky factors.

    Returns:
        (B, N, K) log probabilities.
    """
    # diff: (B, N, K, 2)
    diff = x.unsqueeze(2) - mu.unsqueeze(1)
    B, N, K, _ = diff.shape

    # Solve L @ v = diff via triangular solve
    L_exp = L.unsqueeze(1).expand(B, N, K, 2, 2)
    diff_exp = diff.unsqueeze(-1)  # (B, N, K, 2, 1)
    v = torch.linalg.solve_triangular(L_exp, diff_exp, upper=False)
    mahal = v.squeeze(-1).pow(2).sum(dim=-1)  # (B, N, K)

    # Log determinant of covariance = 2 * sum(log(diag(L)))
    log_det = 2.0 * L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)  # (B, K)
    log_det = log_det.unsqueeze(1)  # (B, 1, K)

    return -0.5 * (2.0 * _LOG2PI + log_det + mahal)


class MDNLoss(nn.Module):
    """Poisson process NLL for mixture density tornado model.

    Loss = Λ - Σ_n log(Σ_k π_k · N(x_n | μ_k, Σ_k))

    where Λ = Σ_k π_k is the total expected count.
    First term penalizes overprediction; second rewards placing
    components near GT locations.
    """

    def __init__(self, lambda_ef: float = 1.0) -> None:
        super().__init__()
        self.lambda_ef = lambda_ef

    def forward(self, mdn_params: dict[str, torch.Tensor],
                tracks: torch.Tensor,
                track_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            mdn_params: dict from MDNHead with pi, mu, L, ef_logits.
            tracks: (B, max_N, 6) GT tracks (cols 0-1 are coords, col 5 is EF).
            track_mask: (B, max_N) bool mask for real tracks.

        Returns:
            Dict with keys: total, spatial, count, ef.
        """
        pi = mdn_params["pi"]          # (B, K)
        mu = mdn_params["mu"]          # (B, K, 2)
        L = mdn_params["L"]            # (B, K, 2, 2)
        ef_logits = mdn_params["ef_logits"]  # (B, K, n_ef)

        B = pi.shape[0]
        device = pi.device

        # Total expected count: Λ = Σ_k π_k
        Lambda = pi.sum(dim=-1)  # (B,)
        count_loss = Lambda.mean()

        # Spatial NLL and EF loss -- batched where possible
        spatial_loss = torch.tensor(0.0, device=device)
        ef_loss = torch.tensor(0.0, device=device)
        n_total = 0

        for b in range(B):
            n = track_mask[b].sum().item()
            if n == 0:
                continue

            gt_coords = tracks[b, :n, :2].unsqueeze(0)  # (1, n, 2)
            gt_ef = tracks[b, :n, 5].long()  # (n,)

            # log N(x_n | μ_k, Σ_k) for all n, k
            log_gauss = _gaussian_log_prob(
                gt_coords, mu[b:b+1], L[b:b+1],
            ).squeeze(0)  # (n, K)

            # log(Σ_k π_k · N(x_n | μ_k, Σ_k))
            log_pi = pi[b].log()  # (K,)
            log_mix = torch.logsumexp(log_pi + log_gauss, dim=-1)  # (n,)
            spatial_loss = spatial_loss - log_mix.sum()

            # EF: weighted by component responsibilities
            with torch.no_grad():
                resp = torch.softmax(log_pi + log_gauss, dim=-1)  # (n, K)

            K = pi.shape[1]
            for k in range(K):
                w = resp[:, k]  # (n,)
                if w.sum() < 1e-8:
                    continue
                ce = F.cross_entropy(
                    ef_logits[b, k].unsqueeze(0).expand(n, -1),
                    gt_ef, reduction="none",
                )
                ef_loss = ef_loss + (w * ce).sum()

            n_total += n

        n_total = max(n_total, 1)
        spatial_loss = spatial_loss / n_total
        ef_loss = ef_loss / n_total

        total = count_loss + spatial_loss + self.lambda_ef * ef_loss

        return {
            "total": total,
            "spatial": spatial_loss,
            "count": count_loss,
            "ef": ef_loss,
        }
