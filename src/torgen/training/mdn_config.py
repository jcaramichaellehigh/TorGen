# src/torgen/training/mdn_config.py
from dataclasses import dataclass


@dataclass
class MDNTrainConfig:
    # Required paths
    drive_dir: str
    checkpoint_dir: str

    # Local cache
    local_cache_dir: str = "/tmp/torgen_mdn"

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    # Training
    batch_size: int = 32
    max_epochs: int = 200
    patience: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-3
    warmup_epochs: int = 5
    kl_anneal_epochs: int = 40
    kl_weight: float = 1.0
    kl_free_bits: float = 0.0
    checkpoint_every: int = 10

    # Model architecture
    d_model: int = 256
    d_z_channel: int = 16
    latent_spatial_size: int = 4
    d_compress: int = 64
    in_channels: int = 16
    n_components: int = 20
    n_ef_classes: int = 6

    # Loss
    lambda_ef: float = 1.0
    lambda_count: float = 1.0

    # Regularization
    dropout: float = 0.1
    env_dropout: float = 0.0  # drop env_vector during training to force z usage

    # Evaluation
    n_eval_samples: int = 10
