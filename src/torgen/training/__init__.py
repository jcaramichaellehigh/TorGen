from torgen.training.config import TrainConfig


def train(config: TrainConfig):
    """Train the TorGen CVAE. Returns the trainer instance."""
    from torgen.training.trainer import Trainer
    trainer = Trainer(config)
    trainer.fit()
    return trainer
