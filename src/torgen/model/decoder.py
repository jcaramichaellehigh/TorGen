import torch
import torch.nn as nn


class TrackDecoderLayer(nn.Module):
    """One DETR decoder layer: self-attn, cross-attn, FFN."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, queries: torch.Tensor,
                memory: torch.Tensor) -> torch.Tensor:
        # Self-attention among queries
        h = self.norm1(queries)
        h2, _ = self.self_attn(h, h, h)
        queries = queries + h2

        # Cross-attention to memory (spatial weather tokens + z token)
        h = self.norm2(queries)
        h2, _ = self.cross_attn(h, memory, memory)
        queries = queries + h2

        # FFN
        h = self.norm3(queries)
        queries = queries + self.ffn(h)
        return queries


class TrackDecoder(nn.Module):
    """DETR-style set prediction decoder.

    350 learned queries cross-attend to spatial weather tokens + a latent z token.
    Output heads: exists (sigmoid), coords (sigmoid), width (sigmoid), ef (logits).
    """

    def __init__(self, num_queries: int = 350, d_model: int = 256,
                 d_latent: int = 64, n_layers: int = 4, n_heads: int = 4,
                 n_ef_classes: int = 6) -> None:
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.z_proj = nn.Linear(d_latent, d_model)
        self.layers = nn.ModuleList(
            [TrackDecoderLayer(d_model, n_heads) for _ in range(n_layers)]
        )
        self.exists_head = nn.Linear(d_model, 1)
        self.coords_head = nn.Linear(d_model, 4)
        self.width_head = nn.Linear(d_model, 1)
        self.ef_head = nn.Linear(d_model, n_ef_classes)

    def forward(self, spatial_tokens: torch.Tensor,
                z: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            spatial_tokens: (B, S, d_model) from CNN encoder
            z: (B, d_latent) latent code

        Returns:
            dict with keys: exists (B,Q,1), coords (B,Q,4),
                           width (B,Q,1), ef_logits (B,Q,6)
        """
        B = spatial_tokens.shape[0]

        # Project z to a single token and append to memory
        z_token = self.z_proj(z).unsqueeze(1)  # (B, 1, d_model)
        memory = torch.cat([spatial_tokens, z_token], dim=1)  # (B, S+1, d_model)

        # Expand learned queries to batch
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        for layer in self.layers:
            queries = layer(queries, memory)

        return {
            "exists": torch.sigmoid(self.exists_head(queries)),
            "coords": torch.sigmoid(self.coords_head(queries)),
            "width": torch.sigmoid(self.width_head(queries)),
            "ef_logits": self.ef_head(queries),  # raw logits for CE loss
        }
