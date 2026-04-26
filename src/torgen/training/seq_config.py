# src/torgen/training/seq_config.py
from dataclasses import dataclass


@dataclass
class SeqTrainConfig:
    # Required paths
    drive_dir: str
    checkpoint_dir: str

    # Local cache
    local_cache_dir: str = "/tmp/torgen_seq"

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    # Training
    batch_size: int = 32
    max_epochs: int = 200
    patience: int = 15
    lr: float = 1e-4
    weight_decay: float = 1e-3
    warmup_epochs: int = 5
    kl_anneal_epochs: int = 40
    checkpoint_every: int = 10

    # Model architecture
    max_seq_len: int = 350
    d_model: int = 256
    d_z_channel: int = 16
    latent_spatial_size: int = 4
    d_compress: int = 64
    n_decoder_layers: int = 4
    n_heads: int = 4
    in_channels: int = 16
    grid_h: int = 270
    grid_w: int = 270
    track_dim: int = 6
    n_ef_classes: int = 6

    # Loss weights
    lambda_coord: float = 5.0
    lambda_bearing: float = 2.0
    lambda_length: float = 2.0
    lambda_width: float = 2.0
    lambda_ef: float = 2.0
    lambda_stop: float = 1.0

    # EF class weight power (0 = uniform, 1 = full inverse-frequency)
    ef_weight_power: float = 0.5

    # Generation
    stop_threshold: float = 0.5

    # Regularization
    dropout: float = 0.1

    # Evaluation
    n_eval_samples: int = 10
