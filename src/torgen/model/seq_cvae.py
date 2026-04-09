# src/torgen/model/seq_cvae.py
"""CVAE with autoregressive sequence decoder for tornado track generation."""
import torch
import torch.nn as nn

from torgen.model.encoder import WeatherEncoder, SpatialCompressor
from torgen.model.seq_decoder import TrackSequenceDecoder
from torgen.model.track_encoder import PosteriorTrackEncoder
from torgen.model.vae import SpatialPrior, SpatialPosterior, reparameterize


class TorGenSeqCVAE(nn.Module):
    """CVAE with autoregressive sequence decoder.

    Same encoder/VAE as TorGenCVAE, but replaces the DETR TrackDecoder
    with a causal TrackSequenceDecoder.
    """

    def __init__(
        self,
        in_channels: int = 16,
        d_model: int = 256,
        d_z_channel: int = 16,
        latent_spatial_size: int = 4,
        d_compress: int = 64,
        max_seq_len: int = 350,
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
        self.decoder = TrackSequenceDecoder(
            d_model=d_model, d_z=d_z_channel,
            latent_spatial_size=latent_spatial_size,
            n_layers=n_decoder_layers, n_heads=n_heads,
            n_ef_classes=n_ef_classes, max_seq_len=max_seq_len,
            dropout=dropout,
        )

    def forward(self, wx: torch.Tensor, tracks: torch.Tensor,
                track_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        spatial_map, _ = self.weather_encoder(wx)
        compressed = self.spatial_compressor(spatial_map)
        track_summary = self.track_encoder(tracks, track_mask)
        mu_q, logvar_q = self.posterior(compressed, track_summary)
        mu_p, logvar_p = self.prior(compressed)
        z = reparameterize(mu_q, logvar_q)  # (B, d_z, 4, 4)
        preds = self.decoder(z, tracks, track_mask)
        return {
            "preds": preds,
            "mu_q": mu_q.flatten(1), "logvar_q": logvar_q.flatten(1),
            "mu_p": mu_p.flatten(1), "logvar_p": logvar_p.flatten(1),
        }

    @torch.no_grad()
    def generate(self, wx: torch.Tensor,
                 stop_threshold: float = 0.5) -> dict[str, torch.Tensor]:
        spatial_map, _ = self.weather_encoder(wx)
        compressed = self.spatial_compressor(spatial_map)
        mu_p, logvar_p = self.prior(compressed)
        z = reparameterize(mu_p, logvar_p)  # (B, d_z, 4, 4)
        preds = self.decoder.generate(z, stop_threshold=stop_threshold)
        return {"preds": preds}
