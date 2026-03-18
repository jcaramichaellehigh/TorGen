import torch
import torch.nn as nn


class TrackDecoderLayer(nn.Module):
    """Self-attention only decoder layer. No cross-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        h = self.norm1(queries)
        h2, _ = self.self_attn(h, h, h)
        queries = queries + self.drop1(h2)

        h = self.norm2(queries)
        queries = queries + self.drop2(self.ffn(h))
        return queries


class TrackDecoder(nn.Module):
    """DETR-style set prediction decoder (z-only, no weather cross-attention).

    350 learned queries conditioned on z via broadcast-add.
    Self-attention among queries for inter-track coordination.
    """

    def __init__(self, num_queries: int = 350, d_model: int = 256,
                 d_latent: int = 64, n_layers: int = 4, n_heads: int = 4,
                 n_ef_classes: int = 6, dropout: float = 0.1) -> None:
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.z_proj = nn.Linear(d_latent, d_model)
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
        B = z.shape[0]

        z_broadcast = self.z_proj(z).unsqueeze(1)
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1) + z_broadcast

        for layer in self.layers:
            queries = layer(queries)

        return {
            "exists": torch.sigmoid(self.exists_head(queries)),
            "coords": torch.sigmoid(self.coords_head(queries)),
            "bearing": torch.sigmoid(self.bearing_head(queries)),
            "length": torch.sigmoid(self.length_head(queries)),
            "width": torch.sigmoid(self.width_head(queries)),
            "ef_logits": self.ef_head(queries),
        }
