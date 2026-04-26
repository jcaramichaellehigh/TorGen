# src/torgen/model/cvae.py
"""Full DETR-CVAE for tornado outbreak generation (v3: spatial latent z)."""
import torch
import torch.nn as nn

from torgen.model.encoder import WeatherEncoder, SpatialCompressor
from torgen.model.decoder import TrackDecoder
from torgen.model.track_encoder import PosteriorTrackEncoder
from torgen.model.vae import SpatialPrior, SpatialPosterior, reparameterize


class TorGenCVAE(nn.Module):
    """Full DETR-CVAE for tornado outbreak generation.

    v3: spatial latent z — compressed weather map feeds prior/posterior,
    decoder cross-attends to flattened z tokens.
    """

    def __init__(
        self,
        in_channels: int = 16,
        d_model: int = 256,
        d_z_channel: int = 16,
        latent_spatial_size: int = 4,
        d_compress: int = 64,
        num_queries: int = 350,
        n_decoder_layers: int = 4,
        n_heads: int = 4,
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
        self.decoder = TrackDecoder(
            num_queries=num_queries, d_model=d_model,
            d_z=d_z_channel, latent_spatial_size=latent_spatial_size,
            n_layers=n_decoder_layers, n_heads=n_heads,
            n_ef_classes=n_ef_classes, dropout=dropout,
        )
        # Count head: weather + pooled z → predicted tornado count (log-rate for Poisson)
        z_pool_dim = d_z_channel * latent_spatial_size ** 2
        self.count_head = nn.Sequential(
            nn.Linear(d_model + z_pool_dim, d_model // 2),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, wx, tracks, track_mask):
        spatial_map, env_vector = self.weather_encoder(wx)
        compressed = self.spatial_compressor(spatial_map)
        track_summary = self.track_encoder(tracks, track_mask)
        mu_q, logvar_q = self.posterior(compressed, track_summary)
        mu_p, logvar_p = self.prior(compressed)
        z = reparameterize(mu_q, logvar_q)  # (B, d_z, 4, 4)
        preds = self.decoder(z, wx_features=spatial_map)
        z_pool = z.flatten(1)  # (B, d_z * H * W)
        count_input = torch.cat([env_vector, z_pool], dim=-1)
        count_log_rate = self.count_head(count_input).squeeze(-1)  # (B,)
        return {
            "preds": preds,
            "count_log_rate": count_log_rate,
            "mu_q": mu_q.flatten(1), "logvar_q": logvar_q.flatten(1),
            "mu_p": mu_p.flatten(1), "logvar_p": logvar_p.flatten(1),
        }

    @torch.no_grad()
    def generate(self, wx):
        spatial_map, env_vector = self.weather_encoder(wx)
        compressed = self.spatial_compressor(spatial_map)
        mu_p, logvar_p = self.prior(compressed)
        z = reparameterize(mu_p, logvar_p)  # (B, d_z, 4, 4)
        preds = self.decoder(z, wx_features=spatial_map)
        z_pool = z.flatten(1)
        count_input = torch.cat([env_vector, z_pool], dim=-1)
        count_log_rate = self.count_head(count_input).squeeze(-1)  # (B,)
        predicted_count = torch.round(count_log_rate.exp()).long()  # (B,)
        return {"preds": preds, "predicted_count": predicted_count}
