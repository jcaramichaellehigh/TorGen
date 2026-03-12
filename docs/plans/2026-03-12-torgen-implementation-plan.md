# TorGen Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the DETR-CVAE tornado outbreak generator as a pip-installable Python package that runs in Google Colab Pro.

**Architecture:** CNN encoder compresses 16-channel weather grids into spatial features. 350 DETR queries cross-attend to weather features + a latent z token to predict variable-length tornado track sets. Hungarian matching loss handles permutation invariance. See `2026-03-11-tornado-cvae-design.md` for full architecture spec and `2026-03-12-torgen-repo-design.md` for repo/workflow spec.

**Tech Stack:** Python 3.11, PyTorch 2.x, scipy, wandb (optional), matplotlib, pandas

**Note:** Tests are run locally during development but NEVER committed to the repo per project conventions.

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/torgen/__init__.py`
- Create: `src/torgen/model/__init__.py`
- Create: `src/torgen/data/__init__.py`
- Create: `src/torgen/loss/__init__.py`
- Create: `src/torgen/training/__init__.py`
- Create: `src/torgen/eval/__init__.py`
- Create: `src/torgen/viz/__init__.py`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "torgen"
version = "0.1.0"
description = "DETR-CVAE for synthetic tornado outbreak generation"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "scipy",
    "pandas",
    "pyarrow",
]

[project.optional-dependencies]
viz = ["matplotlib", "cartopy"]
wandb = ["wandb"]
dev = ["pytest"]
all = ["torgen[viz,wandb,dev]"]

[tool.setuptools.packages.find]
where = ["src"]
```

**Step 2: Create all `__init__.py` files**

All `__init__.py` files are empty initially. Create the full directory tree:

```
src/torgen/__init__.py
src/torgen/model/__init__.py
src/torgen/data/__init__.py
src/torgen/loss/__init__.py
src/torgen/training/__init__.py
src/torgen/eval/__init__.py
src/torgen/viz/__init__.py
```

**Step 3: Verify package installs**

Run: `pip install -e ".[dev]"`
Expected: Installs successfully, `import torgen` works.

**Step 4: Commit**

```bash
git add pyproject.toml src/
git commit -m "scaffold: torgen package structure"
```

---

### Task 2: Training Config

**Files:**
- Create: `src/torgen/training/config.py`
- Test: `tests/test_config.py`

**Step 1: Write failing test**

```python
# tests/test_config.py
from torgen.training.config import TrainConfig


def test_defaults():
    cfg = TrainConfig(
        drive_dir="/data/tensors",
        checkpoint_dir="/data/checkpoints",
    )
    assert cfg.batch_size == 32
    assert cfg.max_epochs == 200
    assert cfg.d_model == 256
    assert cfg.d_latent == 64
    assert cfg.num_queries == 350
    assert cfg.n_decoder_layers == 4
    assert cfg.n_heads == 4
    assert cfg.lr == 1e-4
    assert cfg.seed == 42


def test_override():
    cfg = TrainConfig(
        drive_dir="/data/tensors",
        checkpoint_dir="/data/checkpoints",
        batch_size=16,
        lr=3e-4,
    )
    assert cfg.batch_size == 16
    assert cfg.lr == 3e-4
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — module not found.

**Step 3: Implement config**

```python
# src/torgen/training/config.py
from dataclasses import dataclass


@dataclass
class TrainConfig:
    # Required paths
    drive_dir: str
    checkpoint_dir: str

    # Local cache
    local_cache_dir: str = "/tmp/torgen"

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    # Training
    batch_size: int = 32
    max_epochs: int = 200
    patience: int = 15
    lr: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    kl_anneal_epochs: int = 20
    checkpoint_every: int = 10

    # Model architecture
    num_queries: int = 350
    d_model: int = 256
    d_latent: int = 64
    n_decoder_layers: int = 4
    n_posterior_layers: int = 2
    n_heads: int = 4
    in_channels: int = 16
    grid_h: int = 270
    grid_w: int = 270
    track_dim: int = 6
    n_ef_classes: int = 6

    # Loss weights
    lambda_coord: float = 5.0
    lambda_width: float = 2.0
    lambda_ef: float = 2.0
    lambda_exists: float = 2.0
    lambda_noobj: float = 1.0

    # Generation
    exists_threshold: float = 0.5

    # Tracking
    use_wandb: bool = True
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/torgen/training/config.py
git commit -m "feat: add TrainConfig dataclass"
```

---

### Task 3: Synthetic Data Generator

**Files:**
- Create: `src/torgen/data/synthetic.py`
- Test: `tests/test_synthetic.py`

**Step 1: Write failing test**

```python
# tests/test_synthetic.py
import os
import torch
from torgen.data.synthetic import generate_synthetic_dataset


def test_generates_pt_files(tmp_path):
    generate_synthetic_dataset(
        output_dir=str(tmp_path),
        n_samples=10,
        grid_h=270,
        grid_w=270,
        n_channels=16,
        max_tracks=10,
        seed=42,
    )
    pt_files = list(tmp_path.glob("*.pt"))
    assert len(pt_files) == 10

    sample = torch.load(pt_files[0], weights_only=False)
    assert sample["wx"].shape == (16, 270, 270)
    assert sample["tracks"].ndim == 2
    assert sample["tracks"].shape[1] == 6 or sample["tracks"].shape[0] == 0
    assert "date" in sample


def test_null_days_exist(tmp_path):
    """With enough samples and seed, some days should have 0 tracks."""
    generate_synthetic_dataset(
        output_dir=str(tmp_path),
        n_samples=50,
        grid_h=270,
        grid_w=270,
        n_channels=16,
        max_tracks=5,
        seed=0,
    )
    has_null = False
    for f in tmp_path.glob("*.pt"):
        sample = torch.load(f, weights_only=False)
        if sample["tracks"].shape[0] == 0:
            has_null = True
            break
    assert has_null, "Expected at least one null day in 50 samples"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_synthetic.py -v`
Expected: FAIL — module not found.

**Step 3: Implement synthetic data generator**

```python
# src/torgen/data/synthetic.py
import os
import torch


def generate_synthetic_dataset(
    output_dir: str,
    n_samples: int = 20,
    grid_h: int = 270,
    grid_w: int = 270,
    n_channels: int = 16,
    max_tracks: int = 10,
    seed: int = 42,
) -> None:
    """Generate fake .pt files matching the real data format.

    Each file contains:
        wx: (n_channels, grid_h, grid_w) float32 in [0, 1]
        tracks: (N, 6) float32 — N can be 0 for null days
        date: str YYYY-MM-DD
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = torch.Generator().manual_seed(seed)

    for i in range(n_samples):
        wx = torch.rand(n_channels, grid_h, grid_w, generator=rng)

        # ~30% chance of null day
        if torch.rand(1, generator=rng).item() < 0.3:
            tracks = torch.zeros(0, 6)
        else:
            n_tracks = int(torch.randint(1, max_tracks + 1, (1,), generator=rng).item())
            coords = torch.rand(n_tracks, 4, generator=rng)  # se, sn, ee, en
            width = torch.rand(n_tracks, 1, generator=rng)
            ef = torch.randint(0, 6, (n_tracks, 1), generator=rng).float()
            tracks = torch.cat([coords, width, ef], dim=1)

        date = f"2000-{(i // 28) + 3:02d}-{(i % 28) + 1:02d}"

        torch.save({"wx": wx, "tracks": tracks, "date": date}, os.path.join(output_dir, f"{date}.pt"))
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_synthetic.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/torgen/data/synthetic.py
git commit -m "feat: synthetic data generator for smoke testing"
```

---

### Task 4: Dataset and Collate Function

**Files:**
- Create: `src/torgen/data/dataset.py`
- Test: `tests/test_dataset.py`

**Step 1: Write failing test**

```python
# tests/test_dataset.py
import torch
from torgen.data.synthetic import generate_synthetic_dataset
from torgen.data.dataset import TornadoDataset, tornado_collate


def test_dataset_length(tmp_path):
    generate_synthetic_dataset(str(tmp_path), n_samples=10, seed=42)
    ds = TornadoDataset(str(tmp_path))
    assert len(ds) == 10


def test_dataset_getitem(tmp_path):
    generate_synthetic_dataset(str(tmp_path), n_samples=5, seed=42)
    ds = TornadoDataset(str(tmp_path))
    sample = ds[0]
    assert sample["wx"].shape == (16, 270, 270)
    assert sample["tracks"].ndim == 2
    assert sample["tracks"].shape[1] == 6 or sample["tracks"].shape[0] == 0


def test_collate_pads_tracks(tmp_path):
    generate_synthetic_dataset(str(tmp_path), n_samples=10, seed=42)
    ds = TornadoDataset(str(tmp_path))
    batch = tornado_collate([ds[i] for i in range(4)])

    assert batch["wx"].shape[0] == 4  # batch dim
    assert batch["wx"].shape[1] == 16
    assert batch["tracks"].ndim == 3  # (B, max_N, 6)
    assert batch["track_mask"].shape == batch["tracks"].shape[:2]  # (B, max_N)
    assert batch["track_mask"].dtype == torch.bool


def test_collate_with_null_day(tmp_path):
    """If all days are null, tracks should be (B, 0, 6)."""
    generate_synthetic_dataset(str(tmp_path), n_samples=5, max_tracks=5, seed=0)
    ds = TornadoDataset(str(tmp_path))
    # Force a batch of just null-day samples by finding them
    nulls = [i for i in range(len(ds)) if ds[i]["tracks"].shape[0] == 0]
    if len(nulls) >= 2:
        batch = tornado_collate([ds[i] for i in nulls[:2]])
        assert batch["tracks"].shape == (2, 0, 6)
        assert batch["track_mask"].shape == (2, 0)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL — module not found.

**Step 3: Implement dataset and collate**

```python
# src/torgen/data/dataset.py
import os
from typing import Any

import torch
from torch.utils.data import Dataset


class TornadoDataset(Dataset):
    """Dataset of pre-processed .pt tornado day samples."""

    def __init__(self, data_dir: str, preload: bool = False) -> None:
        self.data_dir = data_dir
        self.files = sorted(
            [f for f in os.listdir(data_dir) if f.endswith(".pt")]
        )
        self.preload = preload
        self._cache: list[dict[str, Any] | None] = [None] * len(self.files)
        if preload:
            for i in range(len(self.files)):
                self._cache[i] = torch.load(
                    os.path.join(data_dir, self.files[i]), weights_only=False
                )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self._cache[idx] is not None:
            return self._cache[idx]
        return torch.load(
            os.path.join(self.data_dir, self.files[idx]), weights_only=False
        )


def tornado_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate variable-length track sets into a padded batch.

    Returns:
        wx: (B, C, H, W)
        tracks: (B, max_N, 6) — zero-padded
        track_mask: (B, max_N) — True for real tracks, False for padding
        dates: list[str]
    """
    wx = torch.stack([s["wx"] for s in samples])
    dates = [s["date"] for s in samples]
    track_counts = [s["tracks"].shape[0] for s in samples]
    max_n = max(track_counts) if track_counts else 0

    B = len(samples)
    tracks = torch.zeros(B, max_n, 6)
    track_mask = torch.zeros(B, max_n, dtype=torch.bool)

    for i, s in enumerate(samples):
        n = s["tracks"].shape[0]
        if n > 0:
            tracks[i, :n] = s["tracks"]
            track_mask[i, :n] = True

    return {"wx": wx, "tracks": tracks, "track_mask": track_mask, "dates": dates}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/torgen/data/dataset.py
git commit -m "feat: TornadoDataset and collate with padding/masking"
```

---

### Task 5: CNN Weather Encoder

**Files:**
- Create: `src/torgen/model/encoder.py`
- Test: `tests/test_encoder.py`

**Step 1: Write failing test**

```python
# tests/test_encoder.py
import torch
from torgen.model.encoder import WeatherEncoder


def test_output_shape():
    enc = WeatherEncoder(in_channels=16, d_model=256)
    x = torch.randn(2, 16, 270, 270)
    spatial_map, env_vector = enc(x)
    # 270 -> 135 -> 68 -> 34 -> 17 (4 stride-2 convs)
    assert spatial_map.shape == (2, 256, 17, 17)
    assert env_vector.shape == (2, 256)


def test_small_grid():
    """Verify it works on a smaller grid (for smoke testing)."""
    enc = WeatherEncoder(in_channels=16, d_model=256)
    x = torch.randn(2, 16, 32, 32)
    spatial_map, env_vector = enc(x)
    assert spatial_map.shape[0] == 2
    assert spatial_map.shape[1] == 256
    assert env_vector.shape == (2, 256)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_encoder.py -v`
Expected: FAIL — module not found.

**Step 3: Implement encoder**

```python
# src/torgen/model/encoder.py
import math

import torch
import torch.nn as nn


class WeatherEncoder(nn.Module):
    """CNN backbone: (B, C_in, H, W) -> spatial feature map + environment vector.

    4 conv blocks with stride-2 downsampling. 2D sinusoidal positional encoding
    added to the spatial feature map. Environment vector obtained by global
    average pooling.
    """

    def __init__(self, in_channels: int = 16, d_model: int = 256) -> None:
        super().__init__()
        channels = [in_channels, 64, 128, d_model]
        blocks: list[nn.Module] = []
        for i in range(len(channels) - 1):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(channels[i + 1]),
                    nn.LeakyReLU(inplace=True),
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
        )
        # Vertical positions
        pos_h = torch.arange(h, device=device, dtype=torch.float).unsqueeze(1)
        pe[0 : d_half : 2, :, :] = torch.sin(pos_h * div_term[:, None]).permute(2, 0, 1).expand(-1, -1, w)
        pe[1 : d_half : 2, :, :] = torch.cos(pos_h * div_term[:, None]).permute(2, 0, 1).expand(-1, -1, w)
        # Horizontal positions
        pos_w = torch.arange(w, device=device, dtype=torch.float).unsqueeze(1)
        pe[d_half : d_half + d_half : 2, :, :] = (
            torch.sin(pos_w * div_term[:, None]).permute(2, 0, 1).expand(-1, h, -1)
        )
        pe[d_half + 1 : d_half + d_half : 2, :, :] = (
            torch.cos(pos_w * div_term[:, None]).permute(2, 0, 1).expand(-1, h, -1)
        )
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_encoder.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/torgen/model/encoder.py
git commit -m "feat: CNN weather encoder with 2D sinusoidal PE"
```

---

### Task 6: VAE Latent Space

**Files:**
- Create: `src/torgen/model/vae.py`
- Test: `tests/test_vae.py`

**Step 1: Write failing test**

```python
# tests/test_vae.py
import torch
from torgen.model.vae import Prior, Posterior, reparameterize


def test_prior_shapes():
    prior = Prior(d_env=256, d_latent=64)
    env = torch.randn(4, 256)
    mu, logvar = prior(env)
    assert mu.shape == (4, 64)
    assert logvar.shape == (4, 64)


def test_posterior_shapes():
    posterior = Posterior(d_env=256, d_track_summary=256, d_latent=64)
    env = torch.randn(4, 256)
    track_summary = torch.randn(4, 256)
    mu, logvar = posterior(env, track_summary)
    assert mu.shape == (4, 64)
    assert logvar.shape == (4, 64)


def test_reparameterize_shape():
    mu = torch.zeros(4, 64)
    logvar = torch.zeros(4, 64)
    z = reparameterize(mu, logvar)
    assert z.shape == (4, 64)


def test_reparameterize_deterministic_at_eval():
    """When logvar is 0, reparameterize should return mu (in expectation)."""
    mu = torch.ones(1000, 64) * 5.0
    logvar = torch.full((1000, 64), -100.0)  # effectively zero variance
    z = reparameterize(mu, logvar)
    assert torch.allclose(z, mu, atol=1e-3)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vae.py -v`
Expected: FAIL — module not found.

**Step 3: Implement VAE components**

```python
# src/torgen/model/vae.py
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vae.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/torgen/model/vae.py
git commit -m "feat: VAE prior, posterior, reparameterization, KL divergence"
```

---

### Task 7: Posterior Track Set Encoder

**Files:**
- Create: `src/torgen/model/track_encoder.py`
- Test: `tests/test_track_encoder.py`

**Step 1: Write failing test**

```python
# tests/test_track_encoder.py
import torch
from torgen.model.track_encoder import PosteriorTrackEncoder


def test_output_shape():
    enc = PosteriorTrackEncoder(track_dim=6, d_model=256, n_layers=2, n_heads=4)
    tracks = torch.randn(2, 10, 6)  # batch=2, 10 tracks each
    track_mask = torch.ones(2, 10, dtype=torch.bool)
    spatial_tokens = torch.randn(2, 289, 256)  # flattened spatial map
    summary = enc(tracks, track_mask, spatial_tokens)
    assert summary.shape == (2, 256)


def test_null_day():
    """If a sample has no tracks, summary should still be a valid tensor."""
    enc = PosteriorTrackEncoder(track_dim=6, d_model=256, n_layers=2, n_heads=4)
    tracks = torch.zeros(2, 0, 6)
    track_mask = torch.zeros(2, 0, dtype=torch.bool)
    spatial_tokens = torch.randn(2, 289, 256)
    summary = enc(tracks, track_mask, spatial_tokens)
    assert summary.shape == (2, 256)
    assert torch.isfinite(summary).all()


def test_variable_length():
    """Masking should handle different track counts within a batch."""
    enc = PosteriorTrackEncoder(track_dim=6, d_model=256, n_layers=2, n_heads=4)
    tracks = torch.randn(2, 5, 6)
    track_mask = torch.zeros(2, 5, dtype=torch.bool)
    track_mask[0, :3] = True   # sample 0 has 3 real tracks
    track_mask[1, :1] = True   # sample 1 has 1 real track
    spatial_tokens = torch.randn(2, 289, 256)
    summary = enc(tracks, track_mask, spatial_tokens)
    assert summary.shape == (2, 256)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_track_encoder.py -v`
Expected: FAIL — module not found.

**Step 3: Implement posterior track encoder**

```python
# src/torgen/model/track_encoder.py
import torch
import torch.nn as nn


class PosteriorTrackEncoderLayer(nn.Module):
    """One layer: self-attention among tracks, cross-attention to weather, FFN."""

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

    def forward(self, track_tokens: torch.Tensor, spatial_tokens: torch.Tensor,
                track_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Self-attention among track tokens
        h = self.norm1(track_tokens)
        h2, _ = self.self_attn(h, h, h, key_padding_mask=track_key_padding_mask)
        track_tokens = track_tokens + h2

        # Cross-attention to spatial weather tokens
        h = self.norm2(track_tokens)
        h2, _ = self.cross_attn(h, spatial_tokens, spatial_tokens)
        track_tokens = track_tokens + h2

        # FFN
        h = self.norm3(track_tokens)
        track_tokens = track_tokens + self.ffn(h)
        return track_tokens


class PosteriorTrackEncoder(nn.Module):
    """Encodes ground truth track set into a summary vector for the posterior.

    Handles null days (0 tracks) by returning a learned null embedding.
    """

    def __init__(self, track_dim: int = 6, d_model: int = 256,
                 n_layers: int = 2, n_heads: int = 4) -> None:
        super().__init__()
        self.embed = nn.Linear(track_dim, d_model)
        self.layers = nn.ModuleList(
            [PosteriorTrackEncoderLayer(d_model, n_heads) for _ in range(n_layers)]
        )
        self.null_embedding = nn.Parameter(torch.randn(d_model))

    def forward(self, tracks: torch.Tensor, track_mask: torch.Tensor,
                spatial_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tracks: (B, N, 6) padded track vectors
            track_mask: (B, N) True for real tracks
            spatial_tokens: (B, S, d_model) flattened spatial feature map

        Returns:
            summary: (B, d_model)
        """
        B = tracks.shape[0]

        # Handle empty track sets
        if tracks.shape[1] == 0:
            return self.null_embedding.unsqueeze(0).expand(B, -1)

        tokens = self.embed(tracks)  # (B, N, d_model)

        # MHA key_padding_mask: True = ignore, so invert our mask
        padding_mask = ~track_mask

        for layer in self.layers:
            tokens = layer(tokens, spatial_tokens, track_key_padding_mask=padding_mask)

        # Mean pool over real tracks only
        mask_expanded = track_mask.unsqueeze(-1).float()  # (B, N, 1)
        counts = mask_expanded.sum(dim=1).clamp(min=1)    # (B, 1)
        summary = (tokens * mask_expanded).sum(dim=1) / counts

        # For any samples that had no real tracks, use null embedding
        no_tracks = ~track_mask.any(dim=1)  # (B,)
        if no_tracks.any():
            summary[no_tracks] = self.null_embedding

        return summary
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_track_encoder.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/torgen/model/track_encoder.py
git commit -m "feat: posterior track set encoder with transformer layers"
```

---

### Task 8: DETR Track Decoder

**Files:**
- Create: `src/torgen/model/decoder.py`
- Test: `tests/test_decoder.py`

**Step 1: Write failing test**

```python
# tests/test_decoder.py
import torch
from torgen.model.decoder import TrackDecoder


def test_output_shapes():
    dec = TrackDecoder(
        num_queries=350, d_model=256, d_latent=64,
        n_layers=4, n_heads=4, n_ef_classes=6,
    )
    spatial_tokens = torch.randn(2, 289, 256)
    z = torch.randn(2, 64)
    out = dec(spatial_tokens, z)

    assert out["exists"].shape == (2, 350, 1)
    assert out["coords"].shape == (2, 350, 4)
    assert out["width"].shape == (2, 350, 1)
    assert out["ef_logits"].shape == (2, 350, 6)


def test_exists_in_zero_one():
    dec = TrackDecoder(
        num_queries=350, d_model=256, d_latent=64,
        n_layers=4, n_heads=4, n_ef_classes=6,
    )
    spatial_tokens = torch.randn(1, 289, 256)
    z = torch.randn(1, 64)
    out = dec(spatial_tokens, z)
    assert (out["exists"] >= 0).all() and (out["exists"] <= 1).all()
    assert (out["coords"] >= 0).all() and (out["coords"] <= 1).all()
    assert (out["width"] >= 0).all() and (out["width"] <= 1).all()


def test_few_queries():
    """Smaller query count for fast smoke tests."""
    dec = TrackDecoder(
        num_queries=8, d_model=64, d_latent=16,
        n_layers=2, n_heads=2, n_ef_classes=6,
    )
    spatial_tokens = torch.randn(2, 16, 64)
    z = torch.randn(2, 16)
    out = dec(spatial_tokens, z)
    assert out["exists"].shape == (2, 8, 1)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_decoder.py -v`
Expected: FAIL — module not found.

**Step 3: Implement decoder**

```python
# src/torgen/model/decoder.py
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_decoder.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/torgen/model/decoder.py
git commit -m "feat: DETR track decoder with output heads"
```

---

### Task 9: Full CVAE Assembly

**Files:**
- Create: `src/torgen/model/cvae.py`
- Test: `tests/test_cvae.py`

**Step 1: Write failing test**

```python
# tests/test_cvae.py
import torch
from torgen.model.cvae import TorGenCVAE


def _small_model():
    return TorGenCVAE(
        in_channels=16, d_model=64, d_latent=16,
        num_queries=8, n_decoder_layers=2, n_posterior_layers=2,
        n_heads=2, n_ef_classes=6,
    )


def test_train_forward():
    model = _small_model()
    model.train()
    wx = torch.randn(2, 16, 32, 32)
    tracks = torch.randn(2, 5, 6)
    track_mask = torch.ones(2, 5, dtype=torch.bool)

    out = model(wx, tracks, track_mask)
    assert "preds" in out
    assert "mu_q" in out and "logvar_q" in out
    assert "mu_p" in out and "logvar_p" in out
    assert out["preds"]["exists"].shape[1] == 8


def test_generate():
    model = _small_model()
    model.eval()
    wx = torch.randn(2, 16, 32, 32)

    with torch.no_grad():
        out = model.generate(wx)
    assert "preds" in out
    assert "mu_q" not in out  # no posterior at generation time
    assert out["preds"]["exists"].shape[0] == 2


def test_generate_returns_different_samples():
    """Different calls to generate should produce different outputs (stochastic z)."""
    model = _small_model()
    model.eval()
    wx = torch.randn(1, 16, 32, 32)

    with torch.no_grad():
        out1 = model.generate(wx)
        out2 = model.generate(wx)
    # Exists probs should differ because z is sampled differently
    assert not torch.allclose(out1["preds"]["exists"], out2["preds"]["exists"])
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cvae.py -v`
Expected: FAIL — module not found.

**Step 3: Implement CVAE**

```python
# src/torgen/model/cvae.py
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
    ) -> None:
        super().__init__()
        self.weather_encoder = WeatherEncoder(in_channels, d_model)
        self.track_encoder = PosteriorTrackEncoder(
            track_dim=6, d_model=d_model,
            n_layers=n_posterior_layers, n_heads=n_heads,
        )
        self.prior = Prior(d_env=d_model, d_latent=d_latent)
        self.posterior = Posterior(
            d_env=d_model, d_track_summary=d_model, d_latent=d_latent,
        )
        self.decoder = TrackDecoder(
            num_queries=num_queries, d_model=d_model,
            d_latent=d_latent, n_layers=n_decoder_layers,
            n_heads=n_heads, n_ef_classes=n_ef_classes,
        )

    def forward(
        self,
        wx: torch.Tensor,
        tracks: torch.Tensor,
        track_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
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
    def generate(self, wx: torch.Tensor) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cvae.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/torgen/model/cvae.py
git commit -m "feat: full CVAE assembly with train and generate paths"
```

---

### Task 10: Hungarian Matching Loss

**Files:**
- Create: `src/torgen/loss/hungarian.py`
- Test: `tests/test_hungarian.py`

**Step 1: Write failing test**

```python
# tests/test_hungarian.py
import torch
from torgen.loss.hungarian import HungarianMatchingLoss


def _make_loss():
    return HungarianMatchingLoss(
        lambda_coord=5.0, lambda_width=2.0, lambda_ef=2.0,
        lambda_exists=2.0, lambda_noobj=1.0, n_ef_classes=6,
    )


def test_perfect_prediction():
    """If preds exactly match ground truth, loss should be small."""
    loss_fn = _make_loss()
    B, Q = 1, 8
    # One ground truth track
    tracks = torch.tensor([[[0.5, 0.5, 0.6, 0.6, 0.1, 2.0]]])  # (1, 1, 6)
    track_mask = torch.ones(1, 1, dtype=torch.bool)

    # Construct preds where slot 0 matches perfectly
    exists = torch.zeros(B, Q, 1)
    exists[0, 0] = 1.0
    coords = torch.zeros(B, Q, 4)
    coords[0, 0] = torch.tensor([0.5, 0.5, 0.6, 0.6])
    width = torch.zeros(B, Q, 1)
    width[0, 0] = 0.1
    ef_logits = torch.full((B, Q, 6), -10.0)
    ef_logits[0, 0, 2] = 10.0  # high confidence for EF2

    preds = {"exists": exists, "coords": coords, "width": width, "ef_logits": ef_logits}
    result = loss_fn(preds, tracks, track_mask)
    assert "total" in result
    assert result["total"].item() > 0  # some loss from unmatched slots BCE


def test_null_day():
    """If no ground truth tracks, only noobj loss should contribute."""
    loss_fn = _make_loss()
    B, Q = 1, 8
    tracks = torch.zeros(1, 0, 6)
    track_mask = torch.zeros(1, 0, dtype=torch.bool)

    exists = torch.full((B, Q, 1), 0.5)
    coords = torch.rand(B, Q, 4)
    width = torch.rand(B, Q, 1)
    ef_logits = torch.randn(B, Q, 6)

    preds = {"exists": exists, "coords": coords, "width": width, "ef_logits": ef_logits}
    result = loss_fn(preds, tracks, track_mask)
    assert result["total"].item() > 0
    assert result["coord"].item() == 0.0  # no matched pairs


def test_loss_is_differentiable():
    loss_fn = _make_loss()
    B, Q = 2, 8
    tracks = torch.randn(B, 3, 6).abs().clamp(0, 1)
    tracks[:, :, 5] = torch.randint(0, 6, (B, 3)).float()
    track_mask = torch.ones(B, 3, dtype=torch.bool)

    exists = torch.randn(B, Q, 1).sigmoid().requires_grad_(True)
    coords = torch.randn(B, Q, 4).sigmoid().requires_grad_(True)
    width = torch.randn(B, Q, 1).sigmoid().requires_grad_(True)
    ef_logits = torch.randn(B, Q, 6).requires_grad_(True)

    preds = {"exists": exists, "coords": coords, "width": width, "ef_logits": ef_logits}
    result = loss_fn(preds, tracks, track_mask)
    result["total"].backward()
    assert exists.grad is not None
    assert coords.grad is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_hungarian.py -v`
Expected: FAIL — module not found.

**Step 3: Implement Hungarian matching loss**

```python
# src/torgen/loss/hungarian.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class HungarianMatchingLoss(nn.Module):
    """Permutation-invariant loss with Hungarian matching.

    Matches predicted track slots to ground truth tracks using a cost matrix,
    then computes component losses on matched/unmatched pairs.
    """

    def __init__(
        self,
        lambda_coord: float = 5.0,
        lambda_width: float = 2.0,
        lambda_ef: float = 2.0,
        lambda_exists: float = 2.0,
        lambda_noobj: float = 1.0,
        n_ef_classes: int = 6,
        ef_class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_width = lambda_width
        self.lambda_ef = lambda_ef
        self.lambda_exists = lambda_exists
        self.lambda_noobj = lambda_noobj
        self.n_ef_classes = n_ef_classes
        if ef_class_weights is not None:
            self.register_buffer("ef_class_weights", ef_class_weights)
        else:
            self.ef_class_weights = None

    @torch.no_grad()
    def _match(
        self,
        pred_exists: torch.Tensor,
        pred_coords: torch.Tensor,
        pred_width: torch.Tensor,
        pred_ef_logits: torch.Tensor,
        gt_tracks: torch.Tensor,
    ) -> tuple[list[int], list[int]]:
        """Compute cost matrix and return optimal (pred_idx, gt_idx) pairs."""
        Q = pred_exists.shape[0]
        N = gt_tracks.shape[0]
        if N == 0:
            return [], []

        gt_coords = gt_tracks[:, :4]
        gt_width = gt_tracks[:, 4:5]
        gt_ef = gt_tracks[:, 5].long()

        # Cost components
        cost_coord = torch.cdist(pred_coords, gt_coords, p=1)  # (Q, N)
        cost_width = torch.cdist(pred_width, gt_width, p=1)     # (Q, N)

        # CE cost: for each (pred, gt) pair
        log_probs = F.log_softmax(pred_ef_logits, dim=-1)  # (Q, 6)
        cost_ef = -log_probs[:, gt_ef]  # (Q, N)

        # Exists cost: how far pred_exists is from 1.0
        cost_exists = -torch.log(pred_exists.squeeze(-1) + 1e-8).unsqueeze(1).expand(-1, N)

        cost = (
            self.lambda_coord * cost_coord
            + self.lambda_width * cost_width
            + self.lambda_ef * cost_ef
            + self.lambda_exists * cost_exists
        )

        pred_idx, gt_idx = linear_sum_assignment(cost.cpu().numpy())
        return list(pred_idx), list(gt_idx)

    def forward(
        self,
        preds: dict[str, torch.Tensor],
        tracks: torch.Tensor,
        track_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            preds: dict with exists (B,Q,1), coords (B,Q,4), width (B,Q,1), ef_logits (B,Q,6)
            tracks: (B, N, 6) padded ground truth
            track_mask: (B, N) bool

        Returns:
            dict with total, coord, width, ef, exists, noobj losses
        """
        B = tracks.shape[0]
        device = preds["exists"].device

        total_coord = torch.tensor(0.0, device=device)
        total_width = torch.tensor(0.0, device=device)
        total_ef = torch.tensor(0.0, device=device)
        total_exists = torch.tensor(0.0, device=device)
        total_noobj = torch.tensor(0.0, device=device)
        n_matched = 0
        n_unmatched = 0

        for b in range(B):
            n_gt = track_mask[b].sum().item()
            gt = tracks[b, :n_gt]  # (n_gt, 6)

            pred_exists = preds["exists"][b]      # (Q, 1)
            pred_coords = preds["coords"][b]      # (Q, 4)
            pred_width = preds["width"][b]         # (Q, 1)
            pred_ef_logits = preds["ef_logits"][b] # (Q, 6)

            pred_idx, gt_idx = self._match(
                pred_exists, pred_coords, pred_width, pred_ef_logits, gt
            )

            Q = pred_exists.shape[0]
            matched_set = set(pred_idx)
            unmatched_idx = [i for i in range(Q) if i not in matched_set]

            # Matched losses
            if len(pred_idx) > 0:
                pi = torch.tensor(pred_idx, device=device)
                gi = torch.tensor(gt_idx, device=device)

                total_coord = total_coord + F.l1_loss(
                    pred_coords[pi], gt[:, :4][gi], reduction="sum"
                )
                total_width = total_width + F.l1_loss(
                    pred_width[pi], gt[:, 4:5][gi], reduction="sum"
                )
                gt_ef_classes = gt[:, 5][gi].long()
                total_ef = total_ef + F.cross_entropy(
                    pred_ef_logits[pi], gt_ef_classes,
                    weight=self.ef_class_weights, reduction="sum"
                )
                total_exists = total_exists + F.binary_cross_entropy(
                    pred_exists[pi].squeeze(-1),
                    torch.ones(len(pred_idx), device=device),
                    reduction="sum",
                )
                n_matched += len(pred_idx)

            # Unmatched losses (exists -> 0)
            if len(unmatched_idx) > 0:
                ui = torch.tensor(unmatched_idx, device=device)
                total_noobj = total_noobj + F.binary_cross_entropy(
                    pred_exists[ui].squeeze(-1),
                    torch.zeros(len(unmatched_idx), device=device),
                    reduction="sum",
                )
                n_unmatched += len(unmatched_idx)

        # Normalize by counts (avoid div by zero)
        n_matched = max(n_matched, 1)
        n_unmatched = max(n_unmatched, 1)

        coord_loss = self.lambda_coord * total_coord / n_matched
        width_loss = self.lambda_width * total_width / n_matched
        ef_loss = self.lambda_ef * total_ef / n_matched
        exists_loss = self.lambda_exists * total_exists / n_matched
        noobj_loss = self.lambda_noobj * total_noobj / n_unmatched

        total = coord_loss + width_loss + ef_loss + exists_loss + noobj_loss

        return {
            "total": total,
            "coord": coord_loss,
            "width": width_loss,
            "ef": ef_loss,
            "exists": exists_loss,
            "noobj": noobj_loss,
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_hungarian.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/torgen/loss/hungarian.py
git commit -m "feat: Hungarian matching loss with component tracking"
```

---

### Task 11: Trainer

**Files:**
- Create: `src/torgen/training/trainer.py`
- Test: `tests/test_trainer.py`

**Step 1: Write failing test**

```python
# tests/test_trainer.py
import os
import torch
from torgen.training.config import TrainConfig
from torgen.training.trainer import Trainer
from torgen.data.synthetic import generate_synthetic_dataset


def test_trainer_runs_one_epoch(tmp_path):
    data_dir = str(tmp_path / "data")
    ckpt_dir = str(tmp_path / "checkpoints")
    generate_synthetic_dataset(data_dir, n_samples=10, seed=42)

    cfg = TrainConfig(
        drive_dir=data_dir,
        checkpoint_dir=ckpt_dir,
        local_cache_dir=data_dir,  # skip Drive->SSD copy
        batch_size=4,
        max_epochs=1,
        use_wandb=False,
        # Small model for fast test
        d_model=64,
        d_latent=16,
        num_queries=8,
        n_decoder_layers=2,
        n_posterior_layers=2,
        n_heads=2,
    )

    trainer = Trainer(cfg)
    trainer.fit()

    assert trainer.epoch == 1
    assert len(trainer.train_losses) == 1
    assert trainer.train_losses[0] > 0


def test_checkpoint_save_and_resume(tmp_path):
    data_dir = str(tmp_path / "data")
    ckpt_dir = str(tmp_path / "checkpoints")
    generate_synthetic_dataset(data_dir, n_samples=10, seed=42)

    cfg = TrainConfig(
        drive_dir=data_dir,
        checkpoint_dir=ckpt_dir,
        local_cache_dir=data_dir,
        batch_size=4,
        max_epochs=2,
        checkpoint_every=1,
        use_wandb=False,
        d_model=64, d_latent=16, num_queries=8,
        n_decoder_layers=2, n_posterior_layers=2, n_heads=2,
    )

    trainer = Trainer(cfg)
    trainer.fit()
    assert trainer.epoch == 2

    # New trainer should resume from checkpoint
    trainer2 = Trainer(cfg)
    assert trainer2.epoch == 2  # loaded from checkpoint
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_trainer.py -v`
Expected: FAIL — module not found.

**Step 3: Implement trainer**

```python
# src/torgen/training/trainer.py
import json
import logging
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from torgen.data.dataset import TornadoDataset, tornado_collate
from torgen.loss.hungarian import HungarianMatchingLoss
from torgen.model.cvae import TorGenCVAE
from torgen.training.config import TrainConfig

logger = logging.getLogger(__name__)


class Trainer:
    """Training loop with checkpointing, wandb logging, and health monitoring."""

    def __init__(self, config: TrainConfig) -> None:
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._seed_everything(config.seed)
        self._log_hardware()

        # Copy data from Drive to local SSD if needed
        self._prepare_data()

        # Build components
        self.model = self._build_model().to(self.device)
        self.loss_fn = HungarianMatchingLoss(
            lambda_coord=config.lambda_coord,
            lambda_width=config.lambda_width,
            lambda_ef=config.lambda_ef,
            lambda_exists=config.lambda_exists,
            lambda_noobj=config.lambda_noobj,
        )
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.scheduler = self._build_scheduler()

        # State
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

        # Dataloaders
        self.train_loader = self._build_dataloader("train")
        self.val_loader = self._build_dataloader("val")

        # wandb
        self.wandb_run = None
        if config.use_wandb:
            self._init_wandb()

        # Resume from checkpoint if available
        self._load_checkpoint()

    def _seed_everything(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if self.cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _log_hardware(self) -> None:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / 1e9
            logger.info(f"GPU: {gpu_name} ({vram:.1f} GB VRAM)")
        else:
            logger.info("No GPU available — running on CPU")
        ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
        logger.info(f"System RAM: {ram_gb:.1f} GB")

    def _prepare_data(self) -> None:
        """Copy data from Drive to local SSD if not already present."""
        src = self.cfg.drive_dir
        dst = self.cfg.local_cache_dir
        if src == dst:
            return  # Already local (e.g., in tests)
        if os.path.exists(dst) and len(os.listdir(dst)) > 0:
            logger.info(f"Local cache already exists at {dst}, skipping copy")
            return
        logger.info(f"Copying data from {src} to {dst}...")
        os.makedirs(dst, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        logger.info("Data copy complete")

    def _build_model(self) -> TorGenCVAE:
        return TorGenCVAE(
            in_channels=self.cfg.in_channels,
            d_model=self.cfg.d_model,
            d_latent=self.cfg.d_latent,
            num_queries=self.cfg.num_queries,
            n_decoder_layers=self.cfg.n_decoder_layers,
            n_posterior_layers=self.cfg.n_posterior_layers,
            n_heads=self.cfg.n_heads,
            n_ef_classes=self.cfg.n_ef_classes,
        )

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        warmup = LinearLR(
            self.optimizer, start_factor=1e-3, total_iters=self.cfg.warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            self.optimizer, T_max=self.cfg.max_epochs - self.cfg.warmup_epochs,
        )
        return SequentialLR(
            self.optimizer, schedulers=[warmup, cosine],
            milestones=[self.cfg.warmup_epochs],
        )

    def _build_dataloader(self, split: str) -> DataLoader:
        # For now: if split dirs exist, use them. Otherwise use flat dir.
        split_dir = os.path.join(self.cfg.local_cache_dir, split)
        if os.path.isdir(split_dir):
            data_dir = split_dir
        else:
            data_dir = self.cfg.local_cache_dir
        ds = TornadoDataset(data_dir, preload=True)
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=(split == "train"),
            collate_fn=tornado_collate,
            drop_last=(split == "train"),
        )

    def _init_wandb(self) -> None:
        try:
            import wandb
            self.wandb_run = wandb.init(
                project="torgen",
                config=vars(self.cfg),
                resume="allow",
            )
        except Exception as e:
            logger.warning(f"wandb init failed ({e}), continuing without tracking")
            self.wandb_run = None

    def _get_beta(self) -> float:
        """KL annealing: linear ramp 0 -> 1 over kl_anneal_epochs."""
        if self.cfg.kl_anneal_epochs == 0:
            return 1.0
        return min(1.0, self.epoch / self.cfg.kl_anneal_epochs)

    def _train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        beta = self._get_beta()

        for batch in self.train_loader:
            wx = batch["wx"].to(self.device)
            tracks = batch["tracks"].to(self.device)
            track_mask = batch["track_mask"].to(self.device)

            out = self.model(wx, tracks, track_mask)
            losses = self.loss_fn(out["preds"], tracks, track_mask)

            from torgen.model.vae import kl_divergence
            kl = kl_divergence(
                out["mu_q"], out["logvar_q"], out["mu_p"], out["logvar_p"]
            )
            loss = losses["total"] + beta * kl

            self.optimizer.zero_grad()
            loss.backward()

            # Health check: gradient norm
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100.0)
            if grad_norm > 100:
                logger.warning(f"Gradient norm {grad_norm:.1f} > 100")

            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1

            # Health check: NaN
            if not torch.isfinite(loss):
                logger.warning("Loss is NaN/Inf — check learning rate")

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        beta = self._get_beta()

        for batch in self.val_loader:
            wx = batch["wx"].to(self.device)
            tracks = batch["tracks"].to(self.device)
            track_mask = batch["track_mask"].to(self.device)

            out = self.model(wx, tracks, track_mask)
            losses = self.loss_fn(out["preds"], tracks, track_mask)

            from torgen.model.vae import kl_divergence
            kl = kl_divergence(
                out["mu_q"], out["logvar_q"], out["mu_p"], out["logvar_p"]
            )
            loss = losses["total"] + beta * kl
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _save_checkpoint(self, tag: str = "latest") -> None:
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.cfg.checkpoint_dir, f"checkpoint_{tag}.pt")
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_loss": self.best_val_loss,
                "patience_counter": self.patience_counter,
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "config": vars(self.cfg),
            },
            path,
        )
        logger.info(f"Checkpoint saved: {path}")

    def _load_checkpoint(self) -> None:
        latest = os.path.join(self.cfg.checkpoint_dir, "checkpoint_latest.pt")
        if not os.path.exists(latest):
            return
        logger.info(f"Resuming from {latest}")
        ckpt = torch.load(latest, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.epoch = ckpt["epoch"]
        self.best_val_loss = ckpt["best_val_loss"]
        self.patience_counter = ckpt["patience_counter"]
        self.train_losses = ckpt["train_losses"]
        self.val_losses = ckpt["val_losses"]

    def _log_epoch(self, train_loss: float, val_loss: float) -> None:
        beta = self._get_beta()
        logger.info(
            f"Epoch {self.epoch}/{self.cfg.max_epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"beta={beta:.3f} | lr={self.optimizer.param_groups[0]['lr']:.2e}"
        )
        if self.wandb_run is not None:
            import wandb
            wandb.log({
                "epoch": self.epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "beta": beta,
                "lr": self.optimizer.param_groups[0]["lr"],
            })

    def fit(self) -> None:
        """Run the full training loop."""
        logger.info(
            f"Starting training: {self.cfg.max_epochs} epochs, "
            f"batch_size={self.cfg.batch_size}, device={self.device}"
        )
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {n_params:,}")

        start_epoch = self.epoch
        for epoch in range(start_epoch, self.cfg.max_epochs):
            self.epoch = epoch + 1
            t0 = time.time()
            train_loss = self._train_one_epoch()
            val_loss = self._validate()
            epoch_time = time.time() - t0

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self._log_epoch(train_loss, val_loss)
            logger.info(f"  epoch_time={epoch_time:.1f}s")

            self.scheduler.step()

            # Checkpoint
            if self.epoch % self.cfg.checkpoint_every == 0:
                self._save_checkpoint("latest")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint("best")
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.cfg.patience:
                logger.info(
                    f"Early stopping at epoch {self.epoch} "
                    f"(patience {self.cfg.patience} exceeded)"
                )
                break

        self._save_checkpoint("latest")
        logger.info("Training complete")
        if self.wandb_run is not None:
            import wandb
            wandb.finish()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_trainer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/torgen/training/trainer.py
git commit -m "feat: training loop with checkpointing, KL annealing, health monitoring"
```

---

### Task 12: Public `train()` API

**Files:**
- Modify: `src/torgen/training/__init__.py`
- Test: `tests/test_train_api.py`

**Step 1: Write failing test**

```python
# tests/test_train_api.py
from torgen.training import TrainConfig, train
from torgen.data.synthetic import generate_synthetic_dataset


def test_train_function(tmp_path):
    data_dir = str(tmp_path / "data")
    ckpt_dir = str(tmp_path / "checkpoints")
    generate_synthetic_dataset(data_dir, n_samples=10, seed=42)

    config = TrainConfig(
        drive_dir=data_dir,
        checkpoint_dir=ckpt_dir,
        local_cache_dir=data_dir,
        batch_size=4,
        max_epochs=1,
        use_wandb=False,
        d_model=64, d_latent=16, num_queries=8,
        n_decoder_layers=2, n_posterior_layers=2, n_heads=2,
    )
    train(config)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_api.py -v`
Expected: FAIL — `train` not importable.

**Step 3: Implement public API**

```python
# src/torgen/training/__init__.py
from torgen.training.config import TrainConfig
from torgen.training.trainer import Trainer


def train(config: TrainConfig) -> Trainer:
    """Train the TorGen CVAE. Returns the trainer instance."""
    trainer = Trainer(config)
    trainer.fit()
    return trainer
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_api.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/torgen/training/__init__.py
git commit -m "feat: public train() API"
```

---

### Task 13: Smoke Test Script

**Files:**
- Create: `scripts/smoke_test.py`

**Step 1: Write the smoke test**

```python
#!/usr/bin/env python3
"""End-to-end smoke test: synthetic data -> model -> train 3 steps -> generate.

Run: python scripts/smoke_test.py
Exit 0 on success, exit 1 on failure.
"""
import logging
import sys
import tempfile

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    from torgen.data.synthetic import generate_synthetic_dataset
    from torgen.data.dataset import TornadoDataset, tornado_collate
    from torgen.model.cvae import TorGenCVAE
    from torgen.loss.hungarian import HungarianMatchingLoss
    from torgen.model.vae import kl_divergence

    with tempfile.TemporaryDirectory() as tmp:
        # 1. Generate synthetic data
        logger.info("Step 1: Generating synthetic data...")
        generate_synthetic_dataset(tmp, n_samples=12, seed=42)
        logger.info(f"  Generated 12 .pt files in {tmp}")

        # 2. Build dataset + dataloader
        logger.info("Step 2: Building dataset and dataloader...")
        ds = TornadoDataset(tmp, preload=True)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=4, collate_fn=tornado_collate, shuffle=True,
        )
        batch = next(iter(loader))
        logger.info(f"  wx shape: {batch['wx'].shape}")
        logger.info(f"  tracks shape: {batch['tracks'].shape}")
        logger.info(f"  track_mask shape: {batch['track_mask'].shape}")

        # 3. Build model (small for smoke test)
        logger.info("Step 3: Building model...")
        model = TorGenCVAE(
            in_channels=16, d_model=64, d_latent=16,
            num_queries=8, n_decoder_layers=2, n_posterior_layers=2,
            n_heads=2, n_ef_classes=6,
        )
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Model parameters: {n_params:,}")

        loss_fn = HungarianMatchingLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # 4. Train 3 steps
        logger.info("Step 4: Running 3 training steps...")
        model.train()
        for step in range(3):
            batch = next(iter(loader))
            out = model(batch["wx"], batch["tracks"], batch["track_mask"])
            losses = loss_fn(out["preds"], batch["tracks"], batch["track_mask"])
            kl = kl_divergence(
                out["mu_q"], out["logvar_q"], out["mu_p"], out["logvar_p"]
            )
            total = losses["total"] + kl
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            logger.info(
                f"  Step {step + 1}: loss={total.item():.4f} "
                f"(coord={losses['coord'].item():.3f}, "
                f"ef={losses['ef'].item():.3f}, "
                f"kl={kl.item():.3f})"
            )
            assert torch.isfinite(total), f"Loss is not finite at step {step + 1}"

        # 5. Generation pass
        logger.info("Step 5: Running generation pass...")
        model.eval()
        with torch.no_grad():
            gen_out = model.generate(batch["wx"])
        exists = gen_out["preds"]["exists"]
        n_predicted = (exists > 0.5).sum().item()
        logger.info(f"  Exists probs range: [{exists.min():.3f}, {exists.max():.3f}]")
        logger.info(f"  Predicted tracks (threshold=0.5): {n_predicted}")

    logger.info("Smoke test PASSED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"Smoke test FAILED: {e}", exc_info=True)
        sys.exit(1)
```

**Step 2: Run the smoke test**

Run: `python scripts/smoke_test.py`
Expected: All 5 steps pass, exits with code 0.

**Step 3: Commit**

```bash
git add scripts/smoke_test.py
git commit -m "feat: end-to-end smoke test script"
```

---

### Task 14: Evaluation Metrics

**Files:**
- Create: `src/torgen/eval/metrics.py`
- Test: `tests/test_metrics.py`

**Step 1: Write failing test**

```python
# tests/test_metrics.py
import torch
from torgen.eval.metrics import compute_sample_metrics


def test_sample_metrics_basic():
    # 2 samples: one with 3 tracks, one null day
    pred_exists = torch.tensor([
        [[0.9], [0.8], [0.7], [0.1]],   # 3 predicted tracks
        [[0.1], [0.05], [0.02], [0.01]], # 0 predicted tracks
    ])
    gt_counts = [3, 0]

    metrics = compute_sample_metrics(pred_exists, gt_counts, threshold=0.5)
    assert "count_mae" in metrics
    assert "null_day_accuracy" in metrics
    assert metrics["count_mae"] == 0.0  # perfect count prediction
    assert metrics["null_day_accuracy"] == 1.0


def test_sample_metrics_off_by_one():
    pred_exists = torch.tensor([
        [[0.9], [0.8], [0.1], [0.1]],  # 2 predicted
    ])
    gt_counts = [3]
    metrics = compute_sample_metrics(pred_exists, gt_counts, threshold=0.5)
    assert metrics["count_mae"] == 1.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL — module not found.

**Step 3: Implement metrics**

```python
# src/torgen/eval/metrics.py
import torch
import numpy as np


def compute_sample_metrics(
    pred_exists: torch.Tensor,
    gt_counts: list[int],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute sample-level track count metrics.

    Args:
        pred_exists: (B, Q, 1) exists probabilities
        gt_counts: list of length B with ground truth track counts
        threshold: exists threshold

    Returns:
        dict with count_mae, null_day_accuracy
    """
    pred_counts = (pred_exists.squeeze(-1) > threshold).sum(dim=1).cpu().numpy()
    gt_arr = np.array(gt_counts)

    count_mae = float(np.abs(pred_counts - gt_arr).mean())

    # Null day accuracy
    gt_null = gt_arr == 0
    pred_null = pred_counts == 0
    if gt_null.sum() + (~gt_null).sum() > 0:
        null_correct = ((gt_null & pred_null) | (~gt_null & ~pred_null)).sum()
        null_day_accuracy = float(null_correct / len(gt_arr))
    else:
        null_day_accuracy = 1.0

    return {
        "count_mae": count_mae,
        "null_day_accuracy": null_day_accuracy,
    }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/torgen/eval/metrics.py
git commit -m "feat: sample-level evaluation metrics"
```

---

### Task 15: Reference Colab Notebook

**Files:**
- Create: `notebooks/train.ipynb`

**Step 1: Create the notebook**

Create a Jupyter notebook with the following cells:

**Cell 1 (markdown):**
```markdown
# TorGen: DETR-CVAE for Tornado Outbreak Generation
Install the package, mount Drive, and train.
```

**Cell 2 (code):**
```python
!pip install -q git+https://github.com/<user>/Lehigh.git
```

**Cell 3 (code):**
```python
from google.colab import drive
drive.mount("/content/drive")
```

**Cell 4 (code) — optional wandb:**
```python
# Optional: experiment tracking
# Sign up at wandb.ai, get API key from wandb.ai/authorize
try:
    import wandb
    wandb.login()
except ImportError:
    print("wandb not installed, skipping experiment tracking")
```

**Cell 5 (code) — training:**
```python
import logging
logging.basicConfig(level=logging.INFO)

from torgen.training import TrainConfig, train

config = TrainConfig(
    drive_dir="/content/drive/MyDrive/dsci498/tensors",
    checkpoint_dir="/content/drive/MyDrive/dsci498/checkpoints",
)

trainer = train(config)
```

**Cell 6 (code) — quick eval:**
```python
print(f"Final train loss: {trainer.train_losses[-1]:.4f}")
print(f"Final val loss: {trainer.val_losses[-1]:.4f}")
print(f"Best val loss: {trainer.best_val_loss:.4f}")
print(f"Epochs trained: {trainer.epoch}")
```

**Step 2: Verify notebook opens**

Open in Jupyter or just verify the file is valid JSON.

**Step 3: Commit**

```bash
git add notebooks/train.ipynb
git commit -m "feat: reference Colab training notebook"
```

---

## Task Dependency Summary

```
Task 1: Scaffolding
  └── Task 2: Config
       └── Task 3: Synthetic Data
            └── Task 4: Dataset/Collate
       └── Task 5: CNN Encoder
            └── Task 7: Posterior Track Encoder ──┐
       └── Task 6: VAE ──────────────────────────┤
       └── Task 8: DETR Decoder ─────────────────┤
                                                  v
                                    Task 9: CVAE Assembly
                                         │
                              Task 10: Hungarian Loss
                                         │
                              Task 11: Trainer
                                         │
                              Task 12: train() API
                                         │
                        ┌────────────────┼────────────────┐
                  Task 13: Smoke Test   Task 14: Metrics  Task 15: Notebook
```

Tasks 3-8 can be parallelized after Task 1-2 are done. Tasks 5, 6, 7, 8 are independent of each other. Tasks 13, 14, 15 are independent of each other.
