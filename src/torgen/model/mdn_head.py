# src/torgen/model/mdn_head.py
"""Mixture Density Network head for tornado point process."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MDNHead(nn.Module):
    """MLP that outputs mixture parameters for K components.

    Separate count prediction (scalar) and spatial allocation (softmax weights).
    π_k = exp(count_logit) * weight_k, where weights sum to 1.

    Each component also has:
        mu_k    (2)  location mean via sigmoid -> [0, 1]^2
        L_k     (3)  lower-triangular Cholesky factor for full 2x2 covariance
        ef_k    (6)  EF class logits

    Total output: 1 (count) + K * (1 + 2 + 3 + 6) = 1 + K*12
    """

    def __init__(self, d_input: int = 512, d_hidden: int = 256,
                 n_components: int = 20, n_ef_classes: int = 6,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.n_components = n_components
        self.n_ef_classes = n_ef_classes
        # 1 weight + 2 mu + 3 cholesky + n_ef = 12 per component, plus 1 count
        d_out = 1 + n_components * (1 + 2 + 3 + n_ef_classes)
        self.net = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, d_input) concatenated env_vector + z_pooled.

        Returns:
            Dict with keys:
                pi: (B, K) unnormalized weights = exp(count_logit) * softmax(raw_weights)
                mu: (B, K, 2) component means (sigmoid, in [0,1]^2)
                L:  (B, K, 2, 2) lower-triangular Cholesky factors
                ef_logits: (B, K, n_ef_classes) raw EF logits
                count_pred: (B,) predicted expected count = exp(count_logit)
        """
        B = x.shape[0]
        K = self.n_components
        raw = self.net(x)  # (B, 1 + K*12)

        # Count: first output, log-space
        count_logit = raw[:, 0]  # (B,)
        count_pred = torch.exp(count_logit)  # (B,)

        # Per-component params
        comp_raw = raw[:, 1:].view(B, K, -1)  # (B, K, 12)

        # Weights: softmax so they sum to 1, then scale by count
        weights = F.softmax(comp_raw[:, :, 0], dim=-1)  # (B, K)
        pi = count_pred.unsqueeze(-1) * weights  # (B, K)

        mu = torch.sigmoid(comp_raw[:, :, 1:3])  # (B, K, 2)

        # Cholesky: 3 values -> 2x2 lower-triangular
        L_raw = comp_raw[:, :, 3:6]  # (B, K, 3)
        L = torch.zeros(B, K, 2, 2, device=x.device)
        L[:, :, 0, 0] = F.softplus(L_raw[:, :, 0]) + 1e-4
        L[:, :, 1, 0] = L_raw[:, :, 1]
        L[:, :, 1, 1] = F.softplus(L_raw[:, :, 2]) + 1e-4

        ef_logits = comp_raw[:, :, 6:]  # (B, K, n_ef_classes)

        return {
            "pi": pi, "mu": mu, "L": L, "ef_logits": ef_logits,
            "count_pred": count_pred,
        }
