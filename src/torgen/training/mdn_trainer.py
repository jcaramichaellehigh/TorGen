# src/torgen/training/mdn_trainer.py
"""Training loop for MDN-CVAE tornado point process model."""
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from torgen.data.dataset import TornadoDataset, tornado_collate
from torgen.loss.mdn import MDNLoss
from torgen.model.mdn_cvae import TorGenMDN
from torgen.model.vae import kl_divergence
from torgen.training.mdn_config import MDNTrainConfig

logger = logging.getLogger(__name__)


class MDNTrainer:
    """Training loop for TorGenMDN."""

    def __init__(self, config: MDNTrainConfig) -> None:
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._seed_everything(config.seed)
        self._log_hardware()
        self._prepare_data()

        self.model = self._build_model().to(self.device)
        self.loss_fn = MDNLoss(lambda_ef=config.lambda_ef).to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.scheduler = self._build_scheduler()

        self.epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.loss_history: dict[str, list[float]] = {
            "train_total": [], "val_total": [],
            "train_spatial": [], "val_spatial": [],
            "train_count": [], "val_count": [],
            "train_ef": [], "val_ef": [],
            "train_kl": [], "val_kl": [],
            "beta": [], "lr": [],
        }

        self.train_loader = self._build_dataloader("train")
        self.val_loader = self._build_dataloader("val")

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
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU: {gpu_name} ({vram:.1f} GB VRAM)")
        else:
            logger.info("No GPU available — running on CPU")

    def _prepare_data(self) -> None:
        """Copy data from Drive to local SSD, decompressing .gz files."""
        import gzip

        src = self.cfg.drive_dir
        dst = self.cfg.local_cache_dir
        if src == dst:
            return
        if os.path.exists(dst) and len(os.listdir(dst)) > 0:
            test_files = [f for f in os.listdir(dst) if f.endswith(".pt")]
            if test_files:
                try:
                    torch.load(os.path.join(dst, test_files[0]), weights_only=False)
                    logger.info(f"Local cache already exists at {dst}, skipping copy")
                    return
                except Exception:
                    logger.warning(f"Corrupt local cache at {dst}, re-copying")
                    shutil.rmtree(dst)
            else:
                logger.info(f"Local cache already exists at {dst}, skipping copy")
                return
        logger.info(f"Copying data from {src} to {dst}...")
        os.makedirs(dst, exist_ok=True)
        files = [f for f in sorted(os.listdir(src)) if f.endswith((".pt", ".pt.gz"))]
        for i, fname in enumerate(files):
            src_path = os.path.join(src, fname)
            if fname.endswith(".gz"):
                dst_path = os.path.join(dst, fname[:-3])
                with gzip.open(src_path, "rb") as f_in:
                    with open(dst_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                shutil.copy2(src_path, os.path.join(dst, fname))
            if (i + 1) % 500 == 0:
                logger.info(f"  Copied {i + 1}/{len(files)} files")
                sys.stdout.flush()
        logger.info(f"Data copy complete: {len(files)} files")

    def _build_model(self) -> TorGenMDN:
        return TorGenMDN(
            in_channels=self.cfg.in_channels,
            d_model=self.cfg.d_model,
            d_z_channel=self.cfg.d_z_channel,
            latent_spatial_size=self.cfg.latent_spatial_size,
            d_compress=self.cfg.d_compress,
            n_components=self.cfg.n_components,
            n_ef_classes=self.cfg.n_ef_classes,
            dropout=self.cfg.dropout,
            env_dropout=self.cfg.env_dropout,
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
        ds = TornadoDataset(self.cfg.local_cache_dir, preload=False, split=split)
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=(split == "train"),
            collate_fn=tornado_collate,
            drop_last=(split == "train"),
        )

    def _get_beta(self) -> float:
        if self.cfg.kl_weight == 0.0:
            return 0.0
        if self.cfg.kl_anneal_epochs == 0:
            return self.cfg.kl_weight
        return min(
            self.cfg.kl_weight,
            self.cfg.kl_weight * self.epoch / self.cfg.kl_anneal_epochs,
        )

    def _train_one_epoch(self) -> dict[str, float]:
        self.model.train()
        accum = {
            "total": 0.0, "spatial": 0.0, "count": 0.0, "ef": 0.0, "kl": 0.0,
            "grad_norm": 0.0, "grad_clips": 0,
        }
        n_batches = 0
        beta = self._get_beta()
        max_norm = 5.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch} train")
        for batch in pbar:
            wx = batch["wx"].to(self.device)
            tracks = batch["tracks"].to(self.device)
            track_mask = batch["track_mask"].to(self.device)

            out = self.model(wx, tracks, track_mask)
            losses = self.loss_fn(out["mdn_params"], tracks, track_mask)

            kl = kl_divergence(
                out["mu_q"], out["logvar_q"], out["mu_p"], out["logvar_p"],
                free_bits=self.cfg.kl_free_bits,
            )
            loss = losses["total"] + beta * kl

            self.optimizer.zero_grad()
            loss.backward()

            has_nan_grad = any(
                torch.isnan(p.grad).any()
                for p in self.model.parameters()
                if p.grad is not None
            )
            if has_nan_grad:
                logger.warning("NaN gradients detected — skipping optimizer step")
                self.optimizer.zero_grad()
                continue

            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=max_norm,
            )
            accum["grad_norm"] += grad_norm.item()
            if grad_norm > max_norm:
                accum["grad_clips"] += 1

            self.optimizer.step()

            accum["total"] += loss.item()
            accum["spatial"] += losses["spatial"].item()
            accum["count"] += losses["count"].item()
            accum["ef"] += losses["ef"].item()
            accum["kl"] += kl.item()
            n_batches += 1

            pbar.set_postfix(
                total=f"{accum['total']/n_batches:.2f}",
                spatial=f"{accum['spatial']/n_batches:.2f}",
                count=f"{accum['count']/n_batches:.2f}",
                ef=f"{accum['ef']/n_batches:.2f}",
                kl=f"{accum['kl']/n_batches:.2f}",
            )

            if not torch.isfinite(loss):
                logger.warning("Loss is NaN/Inf — check learning rate")

        n = max(n_batches, 1)
        avg_grad = accum["grad_norm"] / n
        clips = accum["grad_clips"]
        logger.info(
            f"  grad_norm avg={avg_grad:.1f} | "
            f"clipped {clips}/{n_batches} batches ({100*clips/n:.0f}%)"
        )
        return {k: v / n for k, v in accum.items()
                if k not in ("grad_norm", "grad_clips")}

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        self.model.eval()
        accum = {"total": 0.0, "spatial": 0.0, "count": 0.0, "ef": 0.0, "kl": 0.0}
        n_batches = 0
        beta = self._get_beta()

        for batch in tqdm(self.val_loader, desc=f"Epoch {self.epoch} val"):
            wx = batch["wx"].to(self.device)
            tracks = batch["tracks"].to(self.device)
            track_mask = batch["track_mask"].to(self.device)

            out = self.model(wx, tracks, track_mask)
            losses = self.loss_fn(out["mdn_params"], tracks, track_mask)

            kl = kl_divergence(
                out["mu_q"], out["logvar_q"], out["mu_p"], out["logvar_p"],
                free_bits=self.cfg.kl_free_bits,
            )
            loss = losses["total"] + beta * kl

            accum["total"] += loss.item()
            accum["spatial"] += losses["spatial"].item()
            accum["count"] += losses["count"].item()
            accum["ef"] += losses["ef"].item()
            accum["kl"] += kl.item()
            n_batches += 1

        n = max(n_batches, 1)
        return {k: v / n for k, v in accum.items()}

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
                "loss_history": self.loss_history,
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
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.epoch = ckpt["epoch"]
        self.best_val_loss = ckpt["best_val_loss"]
        self.patience_counter = ckpt["patience_counter"]
        if "loss_history" in ckpt:
            self.loss_history = ckpt["loss_history"]

    def _log_epoch(self, train_metrics: dict[str, float],
                   val_metrics: dict[str, float]) -> None:
        beta = self._get_beta()
        logger.info(
            f"Epoch {self.epoch}/{self.cfg.max_epochs} | "
            f"train={train_metrics['total']:.4f} | val={val_metrics['total']:.4f} | "
            f"spatial={train_metrics['spatial']:.4f} | "
            f"count={train_metrics['count']:.2f} | "
            f"ef={train_metrics['ef']:.4f} | "
            f"kl={train_metrics['kl']:.4f} | "
            f"beta={beta:.3f} | lr={self.optimizer.param_groups[0]['lr']:.2e}"
        )
        sys.stdout.flush()
        sys.stderr.flush()

    def fit(self) -> None:
        logger.info(
            f"Starting MDN training: {self.cfg.max_epochs} epochs, "
            f"batch_size={self.cfg.batch_size}, device={self.device}, "
            f"n_components={self.cfg.n_components}"
        )
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model parameters: {n_params:,}")
        sys.stdout.flush()
        sys.stderr.flush()

        start_epoch = self.epoch
        for epoch in range(start_epoch, self.cfg.max_epochs):
            self.epoch = epoch + 1
            t0 = time.time()
            train_metrics = self._train_one_epoch()
            val_metrics = self._validate()
            epoch_time = time.time() - t0

            for key in ["total", "spatial", "count", "ef", "kl"]:
                self.loss_history[f"train_{key}"].append(train_metrics[key])
                self.loss_history[f"val_{key}"].append(val_metrics[key])
            self.loss_history["beta"].append(self._get_beta())
            self.loss_history["lr"].append(self.optimizer.param_groups[0]["lr"])
            self._log_epoch(train_metrics, val_metrics)
            logger.info(f"  epoch_time={epoch_time:.1f}s")

            self.scheduler.step()

            if self.epoch % self.cfg.checkpoint_every == 0:
                self._save_checkpoint("latest")
            if val_metrics["total"] < self.best_val_loss:
                self.best_val_loss = val_metrics["total"]
                self.patience_counter = 0
                self._save_checkpoint("best")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.cfg.patience:
                logger.info(
                    f"Early stopping at epoch {self.epoch} "
                    f"(patience {self.cfg.patience} exceeded)"
                )
                break

        self._save_checkpoint("latest")
        logger.info("Training complete")


def train_mdn(config: MDNTrainConfig) -> MDNTrainer:
    """Train the TorGen MDN-CVAE. Returns the trainer instance."""
    trainer = MDNTrainer(config)
    trainer.fit()
    return trainer
