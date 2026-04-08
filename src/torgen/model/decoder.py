import torch
import torch.nn as nn


class TrackDecoderLayer(nn.Module):
    """Decoder layer: self-attention -> cross-attention to z tokens -> FFN."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop_cross = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor,
                z_tokens: torch.Tensor) -> torch.Tensor:
        # Self-attention
        h = self.norm1(queries)
        h2, _ = self.self_attn(h, h, h)
        queries = queries + self.drop1(h2)

        # Cross-attention to z tokens
        h = self.norm_cross(queries)
        h2, _ = self.cross_attn(query=h, key=z_tokens, value=z_tokens)
        queries = queries + self.drop_cross(h2)

        # FFN
        h = self.norm2(queries)
        queries = queries + self.drop2(self.ffn(h))
        return queries


class TrackDecoder(nn.Module):
    """DETR-style set prediction decoder (v3: cross-attention to spatial z tokens).

    350 learned queries attend to flattened spatial z tokens via cross-attention.
    """

    def __init__(self, num_queries: int = 350, d_model: int = 256,
                 d_z: int = 16, latent_spatial_size: int = 4,
                 n_layers: int = 4, n_heads: int = 4,
                 n_ef_classes: int = 6, dropout: float = 0.1) -> None:
        super().__init__()
        n_tokens = latent_spatial_size ** 2
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.z_proj = nn.Linear(d_z, d_model)
        self.z_pos = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)
        self.layers = nn.ModuleList(
            [TrackDecoderLayer(d_model, n_heads, dropout=dropout)
             for _ in range(n_layers)]
        )
        self.exists_head = nn.Linear(d_model, 1)
        self.coords_head = nn.Linear(d_model, 2)
        self.bearing_head = nn.Linear(d_model, 1)
        self.length_head = nn.Linear(d_model, 1)
        self.width_head = nn.Linear(d_model, 1)
        self.ef_head = nn.Linear(d_model, n_ef_classes)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            z: (B, d_z, H, W) spatial latent map
        """
        B = z.shape[0]

        # Flatten spatial z to token sequence: (B, d_z, H, W) -> (B, H*W, d_z) -> (B, H*W, d_model)
        z_flat = z.flatten(2).transpose(1, 2)  # (B, H*W, d_z)
        z_tokens = self.z_proj(z_flat) + self.z_pos  # (B, H*W, d_model)

        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        for layer in self.layers:
            queries = layer(queries, z_tokens)

        return {
            "exists": torch.sigmoid(self.exists_head(queries)),
            "coords": torch.sigmoid(self.coords_head(queries)),
            "bearing": torch.sigmoid(self.bearing_head(queries)),
            "length": torch.sigmoid(self.length_head(queries)),
            "width": torch.sigmoid(self.width_head(queries)),
            "ef_logits": self.ef_head(queries),
        }
