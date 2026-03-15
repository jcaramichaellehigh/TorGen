import math

import torch
import torch.nn as nn


class WeatherEncoder(nn.Module):
    """CNN backbone: (B, C_in, H, W) -> spatial feature map + environment vector.

    4 conv blocks with stride-2 downsampling. 2D sinusoidal positional encoding
    added to the spatial feature map. Environment vector obtained by global
    average pooling.
    """

    def __init__(self, in_channels: int = 16, d_model: int = 256,
                 dropout: float = 0.1) -> None:
        super().__init__()
        channels = [in_channels, 64, 128, 192, d_model]
        blocks: list[nn.Module] = []
        for i in range(len(channels) - 1):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(channels[i + 1]),
                    nn.LeakyReLU(inplace=True),
                    nn.Dropout2d(dropout),
                )
            )
        self.backbone = nn.Sequential(*blocks)
        self.d_model = d_model

    def _positional_encoding(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        """2D sinusoidal positional encoding: (1, d_model, h, w)."""
        d = self.d_model
        pe = torch.zeros(d, h, w, device=device)
        d_half = d // 2
        div_term = torch.exp(
            torch.arange(0, d_half, 2, device=device, dtype=torch.float)
            * -(math.log(10000.0) / d_half)
        )  # shape: (d_half // 2,)
        n_freq = div_term.shape[0]

        # Vertical positions: encode along height
        pos_h = torch.arange(h, device=device, dtype=torch.float)  # (h,)
        # sin/cos table: (h, n_freq)
        sin_h = torch.sin(pos_h[:, None] * div_term[None, :])  # (h, n_freq)
        cos_h = torch.cos(pos_h[:, None] * div_term[None, :])  # (h, n_freq)
        # Assign to pe channels: interleave sin/cos over first d_half channels
        for i in range(n_freq):
            pe[2 * i, :, :] = sin_h[:, i].unsqueeze(1).expand(-1, w)
            pe[2 * i + 1, :, :] = cos_h[:, i].unsqueeze(1).expand(-1, w)

        # Horizontal positions: encode along width
        pos_w = torch.arange(w, device=device, dtype=torch.float)  # (w,)
        sin_w = torch.sin(pos_w[:, None] * div_term[None, :])  # (w, n_freq)
        cos_w = torch.cos(pos_w[:, None] * div_term[None, :])  # (w, n_freq)
        for i in range(n_freq):
            pe[d_half + 2 * i, :, :] = sin_w[:, i].unsqueeze(0).expand(h, -1)
            pe[d_half + 2 * i + 1, :, :] = cos_w[:, i].unsqueeze(0).expand(h, -1)

        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C_in, H, W) weather tensor

        Returns:
            spatial_map: (B, d_model, H', W') with positional encoding added
            env_vector: (B, d_model) global average pooled
        """
        feat = self.backbone(x)
        pe = self._positional_encoding(feat.shape[2], feat.shape[3], feat.device)
        spatial_map = feat + pe
        env_vector = feat.mean(dim=(2, 3))  # global avg pool (before PE)
        return spatial_map, env_vector
