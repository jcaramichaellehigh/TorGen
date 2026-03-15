# src/torgen/model/cvae.py
"""Full DETR-CVAE for tornado outbreak generation.

Assembles the weather encoder, posterior track encoder, prior/posterior
VAE components, and the DETR-style track decoder into a single module
with separate training (forward) and generation (generate) paths.
"""
import torch
import torch.nn as nn

from torgen.model.encoder import WeatherEncoder
from torgen.model.decoder import TrackDecoder
from torgen.model.track_encoder import PosteriorTrackEncoder
from torgen.model.vae import Prior, Posterior, reparameterize


class TorGenCVAE(nn.Module):
    """Full DETR-CVAE for tornado outbreak generation."""

    def __init__(
        self,
        in_channels: int = 16,
        d_model: int = 256,
        d_latent: int = 64,
        num_queries: int = 350,
        n_decoder_layers: int = 4,
        n_posterior_layers: int = 2,
        n_heads: int = 4,
        n_ef_classes: int = 6,
        dropout: float = 0.1,
        memory_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.weather_encoder = WeatherEncoder(
            in_channels, d_model, dropout=dropout,
        )
        self.track_encoder = PosteriorTrackEncoder(
            track_dim=6, d_model=d_model,
            n_layers=n_posterior_layers, n_heads=n_heads,
            dropout=dropout,
        )
        self.prior = Prior(d_env=d_model, d_latent=d_latent, dropout=dropout)
        self.posterior = Posterior(
            d_env=d_model, d_track_summary=d_model, d_latent=d_latent,
            dropout=dropout,
        )
        self.decoder = TrackDecoder(
            num_queries=num_queries, d_model=d_model,
            d_latent=d_latent, n_layers=n_decoder_layers,
            n_heads=n_heads, n_ef_classes=n_ef_classes,
            dropout=dropout, memory_dropout=memory_dropout,
        )

    def forward(
        self,
        wx: torch.Tensor,
        tracks: torch.Tensor,
        track_mask: torch.Tensor,
    ) -> dict:
        """Training forward pass: uses posterior to sample z.

        Args:
            wx: (B, C, H, W)
            tracks: (B, N, 6) padded
            track_mask: (B, N) bool

        Returns:
            dict with keys: preds, mu_q, logvar_q, mu_p, logvar_p
        """
        spatial_map, env_vector = self.weather_encoder(wx)
        B, D, H, W = spatial_map.shape
        spatial_tokens = spatial_map.flatten(2).permute(0, 2, 1)  # (B, H*W, D)

        track_summary = self.track_encoder(tracks, track_mask, spatial_tokens)

        mu_q, logvar_q = self.posterior(env_vector, track_summary)
        mu_p, logvar_p = self.prior(env_vector)
        z = reparameterize(mu_q, logvar_q)

        preds = self.decoder(spatial_tokens, z)

        return {
            "preds": preds,
            "mu_q": mu_q, "logvar_q": logvar_q,
            "mu_p": mu_p, "logvar_p": logvar_p,
        }

    @torch.no_grad()
    def generate(self, wx: torch.Tensor) -> dict:
        """Generation forward pass: uses prior to sample z.

        Args:
            wx: (B, C, H, W)

        Returns:
            dict with keys: preds
        """
        spatial_map, env_vector = self.weather_encoder(wx)
        spatial_tokens = spatial_map.flatten(2).permute(0, 2, 1)

        mu_p, logvar_p = self.prior(env_vector)
        z = reparameterize(mu_p, logvar_p)

        preds = self.decoder(spatial_tokens, z)
        return {"preds": preds}
