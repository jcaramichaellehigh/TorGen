# src/torgen/model/vae.py
"""VAE latent space: prior, posterior, reparameterization, and KL divergence.

v3: Spatial latent z — prior and posterior operate on compressed 4x4 feature maps.
"""
import torch
import torch.nn as nn


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Reparameterization trick: z = mu + std * epsilon."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps


def kl_divergence(mu_q: torch.Tensor, logvar_q: torch.Tensor,
                  mu_p: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
    """KL(q || p) for diagonal Gaussians, summed over latent dims, mean over batch."""
    kl = 0.5 * (
        logvar_p - logvar_q
        + (logvar_q.exp() + (mu_q - mu_p).pow(2)) / logvar_p.exp()
        - 1.0
    )
    return kl.sum(dim=-1).mean()


class SpatialPrior(nn.Module):
    """p(z | weather): compressed spatial map -> spatial mu, logvar."""

    def __init__(self, d_compress: int = 64, d_z: int = 16,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(d_compress, d_compress, 1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.mu_head = nn.Conv2d(d_compress, d_z, 1)
        self.logvar_head = nn.Conv2d(d_compress, d_z, 1)

    def forward(self, compressed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(compressed)
        return self.mu_head(h), self.logvar_head(h)


class SpatialPosterior(nn.Module):
    """q(z | weather, tracks): FiLM-conditioned compressed map -> spatial mu, logvar."""

    def __init__(self, d_compress: int = 64, d_model: int = 256,
                 d_z: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        self.scale_proj = nn.Linear(d_model, d_compress)
        self.shift_proj = nn.Linear(d_model, d_compress)
        self.net = nn.Sequential(
            nn.Conv2d(d_compress, d_compress, 1),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.mu_head = nn.Conv2d(d_compress, d_z, 1)
        self.logvar_head = nn.Conv2d(d_compress, d_z, 1)

    def forward(self, compressed: torch.Tensor,
                track_summary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.scale_proj(track_summary)[:, :, None, None]
        shift = self.shift_proj(track_summary)[:, :, None, None]
        modulated = compressed * (1 + scale) + shift
        h = self.net(modulated)
        return self.mu_head(h), self.logvar_head(h)
