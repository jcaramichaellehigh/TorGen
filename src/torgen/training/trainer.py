# src/torgen/training/trainer.py
"""Training loop with checkpointing, KL annealing, and health monitoring."""
import logging
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from torgen.data.dataset import TornadoDataset, tornado_collate
from torgen.loss.hungarian import HungarianMatchingLoss
from torgen.model.cvae import TorGenCVAE
from torgen.model.vae import kl_divergence
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

    def _prepare_data(self) -> None:
        """Copy data from Drive to local SSD if not already present."""
        src = self.cfg.drive_dir
        dst = self.cfg.local_cache_dir
        if src == dst:
            return
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
            self.optimizer, T_max=max(self.cfg.max_epochs - self.cfg.warmup_epochs, 1),
        )
        return SequentialLR(
            self.optimizer, schedulers=[warmup, cosine],
            milestones=[self.cfg.warmup_epochs],
        )

    def _build_dataloader(self, split: str) -> DataLoader:
        ds = TornadoDataset(self.cfg.local_cache_dir, preload=True, split=split)
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

            kl = kl_divergence(
                out["mu_q"], out["logvar_q"], out["mu_p"], out["logvar_p"]
            )
            loss = losses["total"] + beta * kl

            self.optimizer.zero_grad()
            loss.backward()

            # Skip update if gradients contain NaN (numerical instability)
            has_nan_grad = any(
                torch.isnan(p.grad).any()
                for p in self.model.parameters()
                if p.grad is not None
            )
            if has_nan_grad:
                logger.warning("NaN gradients detected — skipping optimizer step")
                self.optimizer.zero_grad()
                continue

            # Health check: gradient norm (clip at 1.0, standard for transformers)
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            if grad_norm > 10:
                logger.warning(f"Gradient norm {grad_norm:.1f} > 10")

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
