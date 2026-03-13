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
    use_wandb: bool = False
