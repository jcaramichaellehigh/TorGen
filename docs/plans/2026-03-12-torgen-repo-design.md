# Design: TorGen — Repo & Training Workflow

**Date:** 2026-03-12
**Author:** John Caramichael
**Project:** DSCI498 — Synthetic Event Set of Tornado Outbreaks with Generative AI
**Companion doc:** `2026-03-11-tornado-cvae-design.md`

---

## Overview

TorGen is a Python package implementing the DETR-CVAE for synthetic tornado
outbreak generation. The code lives in a GitHub repo and runs in Google Colab
Pro with pre-processed `.pt` data files on Google Drive. The package is
pip-installable from GitHub and exposes a simple API: configure, train, evaluate,
generate.

**Execution environment:** Google Colab Pro (~50 GB RAM, A100/T4 GPU, ~12 hr
runtime limit). Data pipeline (NARR/NLDN/SPC processing) is handled separately;
this repo is model-only.

---

## 1. Repo Structure

```
Lehigh/
├── pyproject.toml                  # Package metadata, dependencies
├── src/
│   └── torgen/
│       ├── __init__.py
│       ├── model/
│       │   ├── __init__.py
│       │   ├── encoder.py          # CNN weather encoder
│       │   ├── decoder.py          # DETR track decoder
│       │   ├── vae.py              # Latent space (prior, posterior, reparam)
│       │   └── cvae.py             # Full CVAE assembly
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset.py          # Dataset class: reads .pt files, collate fn
│       │   ├── sampler.py          # Stratified oversampler by severity tier
│       │   └── synthetic.py        # Fake data generator for smoke testing
│       ├── loss/
│       │   ├── __init__.py
│       │   └── hungarian.py        # Hungarian matching + composite loss
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py          # Training loop, checkpointing, wandb
│       │   └── config.py           # Dataclass-based hyperparameter config
│       ├── eval/
│       │   ├── __init__.py
│       │   └── metrics.py          # Sample-level and catalog-level metrics
│       └── viz/
│           ├── __init__.py
│           └── plots.py            # Notebook-friendly visualization helpers
├── scripts/
│   └── smoke_test.py               # End-to-end smoke test (CPU, no real data)
├── tests/                           # Unit tests (local dev only, not committed)
├── notebooks/
│   └── train.ipynb                  # Reference Colab notebook
└── docs/
    └── plans/
```

### Package Layout

- `src/` layout with `pyproject.toml` — standard Python packaging
- Install from Colab: `!pip install git+https://github.com/<user>/Lehigh.git`
- Clean imports: `from torgen.training import TrainConfig, train`

---

## 2. Dependencies

```
torch >= 2.0
scipy                   # Hungarian matching (linear_sum_assignment)
wandb                   # Experiment tracking (optional, graceful fallback)
matplotlib              # Visualization
cartopy                 # Map projections for track plots
pandas                  # Catalog output
pyarrow                 # Parquet I/O for catalogs
```

No YAML, no Hydra, no heavy config frameworks. Config is a Python dataclass.

---

## 3. Colab Workflow

### 3.1 Notebook API

The reference notebook (`notebooks/train.ipynb`) contains ~10 cells:

```python
# Cell 1: Install
!pip install -q git+https://github.com/<user>/Lehigh.git

# Cell 2: Mount Drive
from google.colab import drive
drive.mount("/content/drive")

# Cell 3: wandb login (optional)
import wandb
wandb.login()

# Cell 4: Configure and train
from torgen.training import TrainConfig, train

config = TrainConfig(
    drive_dir="/content/drive/MyDrive/dsci498/tensors",
    checkpoint_dir="/content/drive/MyDrive/dsci498/checkpoints",
)

train(config)
```

### 3.2 What `train()` Does

1. Detects GPU type, logs hardware info (GPU model, VRAM, RAM)
2. Copies `.pt` files from Drive to local `/tmp/torgen/` SSD
   (skips if already present — survives notebook re-runs within a session)
3. Preloads dataset into RAM (~17 GB fits in Pro's ~50 GB)
4. Builds dataloaders with stratified oversampling
5. Builds model, moves to GPU
6. Initializes wandb if available (logs config, GPU info)
7. Runs training loop:
   - KL annealing (beta ramp 0 -> 1 over 20 epochs)
   - Hungarian matching loss
   - Cosine LR schedule with 5-epoch warm-up
8. Saves checkpoint to Drive every 10 epochs + on best val loss
9. Early stopping (patience 15 on val loss)
10. Logs numerical health warnings (NaN loss, KL collapse, exists mode collapse)

### 3.3 Checkpoint Resume

If the Colab runtime dies, re-run the same notebook. `train()` detects existing
checkpoints in `checkpoint_dir` and resumes from the latest one automatically.
No manual intervention needed.

### 3.4 GPU Handling

- Logs GPU type at startup (A100 vs T4 vs other)
- Default batch size 32; config override available if VRAM is tight on T4
- Sets `torch.backends.cudnn.deterministic = True` for reproducibility
- Seeds: Python, NumPy, PyTorch, CUDA — all set from `config.seed`

---

## 4. Smoke Testing

A synthetic smoke test that validates all code paths without real data or a GPU.

### `torgen.data.synthetic`

Generates fake `.pt` files matching the real format:
- Random weather tensors `(16, 270, 270)`
- Random track sets with 0-10 tracks per sample
- ~20 samples, seconds to generate

### `scripts/smoke_test.py`

1. Generate synthetic `.pt` files to a temp directory
2. Build dataset + dataloader (verifies collate, padding, masking)
3. Instantiate full CVAE model
4. Run 3 training steps on CPU (forward, loss, backward, optimizer step)
5. Run 1 generation pass (prior sampling, exists thresholding)
6. Print shapes and loss values at each step
7. Exit 0 on success, exit 1 with clear error on failure

**Run locally:** `python scripts/smoke_test.py`
**Run in Colab (free tier, no GPU):** `!python -m scripts.smoke_test`

If the smoke test passes, the code is wired correctly. Real data is just
bigger tensors with the same shapes.

---

## 5. Evaluation & Generation

### Post-Training

```python
from torgen.eval import evaluate, generate_catalog

# Sample-level metrics on test set
results = evaluate(config, checkpoint="best")
# -> track count MAE, coord error (km), EF accuracy, null-day accuracy

# Generate synthetic catalog
catalog = generate_catalog(
    config,
    checkpoint="best",
    num_seasons=100,
)
# -> saved to Drive as parquet
# -> columns: date, se, sn, ee, en, width, ef, realization_id
```

### Visualization

```python
from torgen.viz import plot_outbreak, plot_comparison, plot_catalog_diagnostics

plot_outbreak(date="2011-04-27", realization=0)
plot_comparison(date="2011-04-27", num_realizations=4)
plot_catalog_diagnostics(catalog)
```

- `plot_outbreak` — predicted tracks overlaid on CAPE field
- `plot_comparison` — real tracks vs. sampled realizations side by side
- `plot_catalog_diagnostics` — marginal distributions, genesis density map,
  KS test results vs. historical

---

## 6. Numerical Health Monitoring

The trainer logs warnings (to console and wandb) when:

| Condition | Warning |
|-----------|---------|
| Loss is NaN or Inf | "Loss diverged — check learning rate" |
| KL stays near 0 after annealing | "Possible posterior collapse — z is being ignored" |
| All exists probs < 0.1 | "Model predicting no tornadoes — exists mode collapse" |
| All exists probs > 0.9 | "Model predicting tornadoes everywhere — check loss weights" |
| Gradient norm > 100 | "Gradient explosion — consider gradient clipping" |

These don't stop training — they log clearly so you can decide whether to
intervene or let it run.

---

## 7. Recommended Progression

For a first-time GPU user, the path from code to results:

1. **Smoke test on CPU** (local or free Colab) — verify code works, ~30 seconds
2. **Small run** (Colab Pro GPU, 5 epochs, real data) — verify Drive I/O,
   GPU training, checkpointing, wandb logging. ~10 minutes.
3. **Full run** (Colab Pro GPU, 200 epochs with early stopping) — the real
   training. ~2-3 hours on A100.
4. **Evaluation** — metrics on test set, generate catalog, visualize.

Never spend GPU time debugging plumbing.

---

## 8. Configuration Reference

All fields in `TrainConfig` with defaults:

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| drive_dir | str | required | Path to .pt files on mounted Drive |
| checkpoint_dir | str | required | Path for checkpoints on Drive |
| local_cache_dir | str | "/tmp/torgen" | Local SSD cache for data |
| seed | int | 42 | Random seed (Python, NumPy, Torch, CUDA) |
| batch_size | int | 32 | Reduce if VRAM-limited on T4 |
| max_epochs | int | 200 | |
| patience | int | 15 | Early stopping patience |
| lr | float | 1e-4 | |
| weight_decay | float | 1e-5 | |
| warmup_epochs | int | 5 | Linear LR warm-up |
| kl_anneal_epochs | int | 20 | Beta ramp 0 -> 1 |
| checkpoint_every | int | 10 | Epochs between Drive checkpoints |
| num_queries | int | 350 | DETR track query slots |
| d_model | int | 256 | Transformer/encoder hidden dim |
| d_latent | int | 64 | Latent space dimensionality |
| n_decoder_layers | int | 4 | DETR decoder layers |
| n_heads | int | 4 | Attention heads |
| exists_threshold | float | 0.5 | Generation threshold for exists head |
| use_wandb | bool | True | Graceful fallback if unavailable |
| deterministic | bool | True | CUDA deterministic mode |
