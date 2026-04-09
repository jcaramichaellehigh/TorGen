# src/torgen/model/seq_decoder.py
"""Autoregressive causal transformer decoder for tornado track sequences."""
import torch
import torch.nn as nn


class CausalDecoderLayer(nn.Module):
    """Decoder layer: causal self-attention -> cross-attention to z tokens -> FFN."""

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

    def forward(self, x: torch.Tensor, z_tokens: torch.Tensor,
                attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Causal self-attention
        h = self.norm1(x)
        h2, _ = self.self_attn(h, h, h, attn_mask=attn_mask)
        x = x + self.drop1(h2)

        # Cross-attention to z tokens
        h = self.norm_cross(x)
        h2, _ = self.cross_attn(query=h, key=z_tokens, value=z_tokens)
        x = x + self.drop_cross(h2)

        # FFN
        h = self.norm2(x)
        x = x + self.drop2(self.ffn(h))
        return x


class TrackSequenceDecoder(nn.Module):
    """Autoregressive decoder: generates tracks sequentially, cross-attending to z.

    Training: teacher forcing with causal mask (fully parallel).
    Inference: autoregressive loop until P(stop) > threshold or max_seq_len.
    """

    def __init__(self, d_model: int = 256, d_z: int = 16,
                 latent_spatial_size: int = 4, n_layers: int = 4,
                 n_heads: int = 4, n_ef_classes: int = 6,
                 max_seq_len: int = 350, track_dim: int = 6,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        # Input embedding
        self.track_embed = nn.Linear(track_dim, d_model)
        self.start_token = nn.Parameter(torch.randn(d_model) * 0.02)
        self.pos_embed = nn.Embedding(max_seq_len + 1, d_model)  # +1 for START

        # Z token projection (same as DETR decoder)
        n_tokens = latent_spatial_size ** 2
        self.z_proj = nn.Linear(d_z, d_model)
        self.z_pos = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)

        # Transformer layers
        self.layers = nn.ModuleList(
            [CausalDecoderLayer(d_model, n_heads, dropout=dropout)
             for _ in range(n_layers)]
        )

        # Output heads
        self.coords_head = nn.Linear(d_model, 2)
        self.bearing_head = nn.Linear(d_model, 1)
        self.length_head = nn.Linear(d_model, 1)
        self.width_head = nn.Linear(d_model, 1)
        self.ef_head = nn.Linear(d_model, n_ef_classes)
        self.stop_head = nn.Linear(d_model, 1)

    def _prepare_z_tokens(self, z: torch.Tensor) -> torch.Tensor:
        """Flatten spatial z to token sequence: (B, d_z, H, W) -> (B, H*W, d_model)."""
        z_flat = z.flatten(2).transpose(1, 2)  # (B, H*W, d_z)
        return self.z_proj(z_flat) + self.z_pos  # (B, H*W, d_model)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular causal mask: -inf above diagonal."""
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

    def forward(self, z: torch.Tensor, tracks: torch.Tensor,
                track_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """Teacher-forced forward pass.

        Args:
            z: (B, d_z, H, W) spatial latent.
            tracks: (B, max_N, 6) GT tracks in bearing/length format, zero-padded.
            track_mask: (B, max_N) bool mask, True for real tracks.

        Returns:
            Dict with keys: coords, bearing, length, width, ef_logits, stop.
            Each has shape (B, seq_len, ...) where seq_len = max_N + 1
            (one position per GT track + one STOP position).
        """
        B = z.shape[0]
        device = z.device
        z_tokens = self._prepare_z_tokens(z)

        # Build input sequence: [START, track_1, ..., track_n]
        max_n = tracks.shape[1]
        seq_len = max_n + 1  # START + max_N positions

        # Embed GT tracks
        track_tokens = self.track_embed(tracks)  # (B, max_N, d_model)

        # Prepend START token
        start = self.start_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        x = torch.cat([start, track_tokens], dim=1)  # (B, seq_len, d_model)

        # Add positional embeddings
        positions = torch.arange(seq_len, device=device)
        x = x + self.pos_embed(positions)

        # Causal mask
        mask = self._causal_mask(seq_len, device)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, z_tokens, attn_mask=mask)

        # Output heads (applied to all positions)
        return {
            "coords": torch.sigmoid(self.coords_head(x)),
            "bearing": torch.sigmoid(self.bearing_head(x)),
            "length": torch.sigmoid(self.length_head(x)),
            "width": torch.sigmoid(self.width_head(x)),
            "ef_logits": self.ef_head(x),
            "stop": torch.sigmoid(self.stop_head(x)),
        }

    @torch.no_grad()
    def generate(self, z: torch.Tensor,
                 stop_threshold: float = 0.5) -> dict[str, torch.Tensor]:
        """Autoregressive generation.

        Args:
            z: (B, d_z, H, W) spatial latent.
            stop_threshold: stop generating when P(stop) > this.

        Returns:
            Dict with keys: coords, bearing, length, width, ef_logits, stop, gen_mask.
            Each prediction tensor has shape (B, max_seq_len, ...) zero-padded.
            gen_mask: (B, max_seq_len) bool, True for generated tracks.
        """
        B = z.shape[0]
        device = z.device
        z_tokens = self._prepare_z_tokens(z)

        # Collect predictions
        all_coords = []
        all_bearing = []
        all_length = []
        all_width = []
        all_ef_logits = []
        all_stop = []

        # Start with START token
        start = self.start_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        start = start + self.pos_embed(torch.tensor([0], device=device))
        seq = start

        cumulative_active = torch.ones(B, dtype=torch.bool, device=device)

        for step in range(self.max_seq_len):
            # Forward through transformer
            x = seq
            for layer in self.layers:
                x = layer(x, z_tokens)

            # Get prediction from last position
            last = x[:, -1, :]  # (B, d_model)

            coords = torch.sigmoid(self.coords_head(last))        # (B, 2)
            bearing = torch.sigmoid(self.bearing_head(last))       # (B, 1)
            length = torch.sigmoid(self.length_head(last))         # (B, 1)
            width = torch.sigmoid(self.width_head(last))           # (B, 1)
            ef_logits = self.ef_head(last)                         # (B, 6)
            stop_prob = torch.sigmoid(self.stop_head(last))        # (B, 1)

            all_coords.append(coords)
            all_bearing.append(bearing)
            all_length.append(length)
            all_width.append(width)
            all_ef_logits.append(ef_logits)
            all_stop.append(stop_prob)

            # Check stopping
            stopped = stop_prob.squeeze(-1) > stop_threshold
            cumulative_active = cumulative_active & ~stopped
            if not cumulative_active.any():
                break

            # Build next track token from predictions
            ef_float = ef_logits.argmax(dim=-1, keepdim=True).float()
            next_track = torch.cat([coords, bearing, length, width, ef_float], dim=-1)

            # Embed and append to sequence
            next_token = self.track_embed(next_track).unsqueeze(1)
            pos = self.pos_embed(torch.tensor([step + 1], device=device))
            next_token = next_token + pos
            seq = torch.cat([seq, next_token], dim=1)

        # Pad to fixed tensors
        n_generated = len(all_coords)
        pad_len = self.max_seq_len

        def _pad(tensors, feat_dim):
            if not tensors:
                return torch.zeros(B, pad_len, feat_dim, device=device)
            stacked = torch.stack(tensors, dim=1)  # (B, n_generated, feat_dim)
            if n_generated < pad_len:
                pad = torch.zeros(B, pad_len - n_generated, feat_dim, device=device)
                stacked = torch.cat([stacked, pad], dim=1)
            return stacked

        # Rebuild mask from stop decisions
        gen_mask = torch.zeros(B, pad_len, dtype=torch.bool, device=device)
        active = torch.ones(B, dtype=torch.bool, device=device)
        for step_idx in range(n_generated):
            gen_mask[:, step_idx] = active
            stopped = all_stop[step_idx].squeeze(-1) > stop_threshold
            active = active & ~stopped

        return {
            "coords": _pad(all_coords, 2),
            "bearing": _pad(all_bearing, 1),
            "length": _pad(all_length, 1),
            "width": _pad(all_width, 1),
            "ef_logits": _pad(all_ef_logits, 6),
            "stop": _pad(all_stop, 1),
            "gen_mask": gen_mask,
        }
