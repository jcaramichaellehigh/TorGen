# src/torgen/model/mdn_cvae.py
"""CVAE with Mixture Density Network head for tornado point process."""
import torch
import torch.nn as nn

from torgen.model.encoder import WeatherEncoder, SpatialCompressor
from torgen.model.track_encoder import PosteriorTrackEncoder
from torgen.model.vae import SpatialPrior, SpatialPosterior, reparameterize
from torgen.model.mdn_head import MDNHead


class TorGenMDN(nn.Module):
    """CVAE with MDN output for tornado outbreak generation.

    Weather encoder + spatial compressor + VAE produce a latent z.
    MDN head maps (env_vector, z_pooled) to mixture parameters.
    """

    def __init__(
        self,
        in_channels: int = 16,
        d_model: int = 256,
        d_z_channel: int = 16,
        latent_spatial_size: int = 4,
        d_compress: int = 64,
        n_components: int = 20,
        n_ef_classes: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.weather_encoder = WeatherEncoder(
            in_channels, d_model, dropout=dropout,
        )
        self.spatial_compressor = SpatialCompressor(
            d_model=d_model, d_compress=d_compress, dropout=dropout,
        )
        self.track_encoder = PosteriorTrackEncoder(
            track_dim=6, d_model=d_model,
        )
        self.prior = SpatialPrior(
            d_compress=d_compress, d_z=d_z_channel, dropout=dropout,
        )
        self.posterior = SpatialPosterior(
            d_compress=d_compress, d_model=d_model,
            d_z=d_z_channel, dropout=dropout,
        )
        z_pool_dim = d_z_channel * latent_spatial_size ** 2
        self.mdn = MDNHead(
            d_input=d_model + z_pool_dim,
            d_hidden=d_model,
            n_components=n_components,
            n_ef_classes=n_ef_classes,
            dropout=dropout,
        )

    def forward(self, wx: torch.Tensor, tracks: torch.Tensor,
                track_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        spatial_map, env_vector = self.weather_encoder(wx)
        compressed = self.spatial_compressor(spatial_map)
        track_summary = self.track_encoder(tracks, track_mask)
        mu_q, logvar_q = self.posterior(compressed, track_summary)
        mu_p, logvar_p = self.prior(compressed)
        z = reparameterize(mu_q, logvar_q)

        z_pooled = z.flatten(1)
        mdn_input = torch.cat([env_vector, z_pooled], dim=-1)
        mdn_params = self.mdn(mdn_input)

        return {
            "mdn_params": mdn_params,
            "mu_q": mu_q.flatten(1), "logvar_q": logvar_q.flatten(1),
            "mu_p": mu_p.flatten(1), "logvar_p": logvar_p.flatten(1),
        }

    @torch.no_grad()
    def generate(self, wx: torch.Tensor) -> dict[str, torch.Tensor]:
        spatial_map, env_vector = self.weather_encoder(wx)
        compressed = self.spatial_compressor(spatial_map)
        mu_p, logvar_p = self.prior(compressed)
        z = reparameterize(mu_p, logvar_p)

        z_pooled = z.flatten(1)
        mdn_input = torch.cat([env_vector, z_pooled], dim=-1)
        mdn_params = self.mdn(mdn_input)

        return {"mdn_params": mdn_params}
