# src/torgen/model/mdn_head.py
"""Mixture Density Network head for tornado point process."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MDNHead(nn.Module):
    """MLP that outputs mixture parameters for K components.

    Each component has:
        pi_k    (1)  unnormalized weight via softplus (expected count from this region)
        mu_k    (2)  location mean via sigmoid -> [0, 1]^2
        L_k     (3)  lower-triangular Cholesky factor for full 2x2 covariance
        ef_k    (6)  EF class logits

    Total output per component: 12
    """

    def __init__(self, d_input: int = 512, d_hidden: int = 256,
                 n_components: int = 20, n_ef_classes: int = 6,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.n_components = n_components
        self.n_ef_classes = n_ef_classes
        d_out = n_components * (1 + 2 + 3 + n_ef_classes)
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
                pi: (B, K) unnormalized weights (softplus, always >= 0)
                mu: (B, K, 2) component means (sigmoid, in [0,1]^2)
                L:  (B, K, 2, 2) lower-triangular Cholesky factors
                ef_logits: (B, K, n_ef_classes) raw EF logits
        """
        B = x.shape[0]
        K = self.n_components
        raw = self.net(x).view(B, K, -1)  # (B, K, 12)

        pi = F.softplus(raw[:, :, 0])  # (B, K)
        mu = torch.sigmoid(raw[:, :, 1:3])  # (B, K, 2)

        # Cholesky: 3 values -> 2x2 lower-triangular
        L_raw = raw[:, :, 3:6]  # (B, K, 3)
        L = torch.zeros(B, K, 2, 2, device=x.device)
        L[:, :, 0, 0] = F.softplus(L_raw[:, :, 0]) + 1e-4
        L[:, :, 1, 0] = L_raw[:, :, 1]
        L[:, :, 1, 1] = F.softplus(L_raw[:, :, 2]) + 1e-4

        ef_logits = raw[:, :, 6:]  # (B, K, n_ef_classes)

        return {"pi": pi, "mu": mu, "L": L, "ef_logits": ef_logits}
