# src/torgen/model/vae.py
"""VAE latent space: prior, posterior, reparameterization, and KL divergence.

The latent variable z captures stochastic aspects of tornado occurrence
that are not fully determined by the environment alone.
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


class Prior(nn.Module):
    """p(z | weather): environment vector -> mu, logvar."""

    def __init__(self, d_env: int = 256, d_latent: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_env, d_env),
            nn.LeakyReLU(inplace=True),
            nn.Linear(d_env, d_env),
            nn.LeakyReLU(inplace=True),
        )
        self.mu_head = nn.Linear(d_env, d_latent)
        self.logvar_head = nn.Linear(d_env, d_latent)

    def forward(self, env: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(env)
        return self.mu_head(h), self.logvar_head(h)


class Posterior(nn.Module):
    """q(z | weather, tracks): cat(env, track_summary) -> mu, logvar."""

    def __init__(self, d_env: int = 256, d_track_summary: int = 256,
                 d_latent: int = 64) -> None:
        super().__init__()
        d_in = d_env + d_track_summary
        self.net = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.LeakyReLU(inplace=True),
            nn.Linear(d_in, d_env),
            nn.LeakyReLU(inplace=True),
        )
        self.mu_head = nn.Linear(d_env, d_latent)
        self.logvar_head = nn.Linear(d_env, d_latent)

    def forward(self, env: torch.Tensor,
                track_summary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(torch.cat([env, track_summary], dim=-1))
        return self.mu_head(h), self.logvar_head(h)
