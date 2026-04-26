# src/torgen/training/trainer.py
"""Training loop with checkpointing, KL annealing, and health monitoring."""
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
        ef_weights = self._compute_ef_weights(config.ef_weight_power)
        self.loss_fn = HungarianMatchingLoss(
            lambda_coord=config.lambda_coord,
            lambda_bearing=config.lambda_bearing,
            lambda_length=config.lambda_length,
            lambda_width=config.lambda_width,
            lambda_ef=config.lambda_ef,
            lambda_exists=config.lambda_exists,
            focal_gamma=config.focal_gamma,
            ef_class_weights=ef_weights,
            ef_weight_power=config.ef_weight_power,
        ).to(self.device)
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
        self.loss_history: dict[str, list[float]] = {
            "train_total": [], "val_total": [],
            "train_kl": [], "val_kl": [],
            "train_recon": [], "val_recon": [],
            "beta": [], "lr": [],
        }

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
            # Verify at least one file is loadable
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
                dst_path = os.path.join(dst, fname[:-3])  # strip .gz
                with gzip.open(src_path, "rb") as f_in:
                    with open(dst_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                shutil.copy2(src_path, os.path.join(dst, fname))
            if (i + 1) % 500 == 0:
                logger.info(f"  Copied {i + 1}/{len(files)} files")
                sys.stdout.flush()
        logger.info(f"Data copy complete: {len(files)} files")

    def _build_model(self) -> TorGenCVAE:
        return TorGenCVAE(
            in_channels=self.cfg.in_channels,
            d_model=self.cfg.d_model,
            d_z_channel=self.cfg.d_z_channel,
            latent_spatial_size=self.cfg.latent_spatial_size,
            d_compress=self.cfg.d_compress,
            num_queries=self.cfg.num_queries,
            n_decoder_layers=self.cfg.n_decoder_layers,
            n_heads=self.cfg.n_heads,
            n_ef_classes=self.cfg.n_ef_classes,
            dropout=self.cfg.dropout,
        )

    def _compute_ef_weights(self, power: float) -> torch.Tensor | None:
        """Compute inverse-frequency EF class weights from training data.

        Args:
            power: Exponent on inverse frequencies. 0 = uniform, 1 = full IF,
                   0.5 = square-root IF (gentler). Values in between interpolate.

        Returns:
            Tensor of shape (n_ef_classes,), or None if power == 0.
        """
        if power == 0.0:
            return None
        ds = TornadoDataset(self.cfg.local_cache_dir, preload=False, split="train")
        counts = torch.zeros(self.cfg.n_ef_classes)
        for i in range(len(ds)):
            tracks = ds[i]["tracks"]  # (N, 6)
            if tracks.shape[0] > 0:
                ef = tracks[:, 5].long()
                for c in range(self.cfg.n_ef_classes):
                    counts[c] += (ef == c).sum()
        # Avoid division by zero for classes with no samples
        counts = counts.clamp(min=1)
        freq = counts / counts.sum()
        weights = (1.0 / freq) ** power
        weights = weights / weights.mean()  # normalize so mean weight = 1
        logger.info(f"EF class weights (power={power}): {weights.tolist()}")
        return weights

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
        """KL annealing: linear ramp 0 -> kl_weight over kl_anneal_epochs."""
        if self.cfg.kl_weight == 0.0:
            return 0.0
        if self.cfg.kl_anneal_epochs == 0:
            return self.cfg.kl_weight
        return min(self.cfg.kl_weight, self.cfg.kl_weight * self.epoch / self.cfg.kl_anneal_epochs)

    def _train_one_epoch(self) -> dict[str, float]:
        self.model.train()
        accum = {"total": 0.0, "kl": 0.0, "recon": 0.0}
        n_batches = 0
        beta = self._get_beta()

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch} train")
        for batch in pbar:
            wx = batch["wx"].to(self.device)
            tracks = batch["tracks"].to(self.device)
            track_mask = batch["track_mask"].to(self.device)

            out = self.model(wx, tracks, track_mask)
            losses = self.loss_fn(out["preds"], tracks, track_mask)

            kl = kl_divergence(
                out["mu_q"], out["logvar_q"], out["mu_p"], out["logvar_p"],
                free_bits=self.cfg.kl_free_bits,
            )

            # Count loss: L1 on predicted vs actual count
            gt_counts = track_mask.sum(dim=1).float()  # (B,)
            predicted_counts = out["count_log_rate"].exp()  # (B,)
            count_loss = nn.functional.l1_loss(predicted_counts, gt_counts)

            loss = losses["total"] + beta * kl + self.cfg.lambda_count * count_loss

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

            max_norm = 5.0
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
            if grad_norm > max_norm * 4:
                logger.warning(f"Gradient norm {grad_norm:.1f} > {max_norm * 4}")

            self.optimizer.step()

            accum["total"] += loss.item()
            accum["kl"] += kl.item()
            accum["recon"] += losses["total"].item()
            n_batches += 1

            pbar.set_postfix(
                total=f"{accum['total']/n_batches:.2f}",
                recon=f"{accum['recon']/n_batches:.2f}",
                kl=f"{accum['kl']/n_batches:.2f}",
            )

            if not torch.isfinite(loss):
                logger.warning("Loss is NaN/Inf — check learning rate")

        n = max(n_batches, 1)
        return {k: v / n for k, v in accum.items()}

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        self.model.eval()
        accum = {"total": 0.0, "kl": 0.0, "recon": 0.0}
        n_batches = 0
        beta = self._get_beta()

        for batch in tqdm(self.val_loader, desc=f"Epoch {self.epoch} val"):
            wx = batch["wx"].to(self.device)
            tracks = batch["tracks"].to(self.device)
            track_mask = batch["track_mask"].to(self.device)

            out = self.model(wx, tracks, track_mask)
            losses = self.loss_fn(out["preds"], tracks, track_mask)

            kl = kl_divergence(
                out["mu_q"], out["logvar_q"], out["mu_p"], out["logvar_p"],
                free_bits=self.cfg.kl_free_bits,
            )

            gt_counts = track_mask.sum(dim=1).float()
            count_loss = nn.functional.poisson_nll_loss(
                out["count_log_rate"], gt_counts,
                log_input=True, reduction="mean",
            )

            loss = losses["total"] + beta * kl + self.cfg.lambda_count * count_loss

            accum["total"] += loss.item()
            accum["kl"] += kl.item()
            accum["recon"] += losses["total"].item()
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
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
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
        self.train_losses = ckpt["train_losses"]
        self.val_losses = ckpt["val_losses"]
        if "loss_history" in ckpt:
            self.loss_history = ckpt["loss_history"]

    def _log_epoch(self, train_metrics: dict[str, float],
                   val_metrics: dict[str, float]) -> None:
        beta = self._get_beta()
        logger.info(
            f"Epoch {self.epoch}/{self.cfg.max_epochs} | "
            f"train={train_metrics['total']:.4f} | val={val_metrics['total']:.4f} | "
            f"kl={train_metrics['kl']:.4f} | recon={train_metrics['recon']:.4f} | "
            f"beta={beta:.3f} | lr={self.optimizer.param_groups[0]['lr']:.2e}"
        )
        sys.stdout.flush()
        sys.stderr.flush()
        if self.wandb_run is not None:
            import wandb
            wandb.log({
                "epoch": self.epoch,
                "train_loss": train_metrics["total"],
                "val_loss": val_metrics["total"],
                "train_kl": train_metrics["kl"],
                "train_recon": train_metrics["recon"],
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
        sys.stdout.flush()
        sys.stderr.flush()

        start_epoch = self.epoch
        for epoch in range(start_epoch, self.cfg.max_epochs):
            self.epoch = epoch + 1
            t0 = time.time()
            train_metrics = self._train_one_epoch()
            val_metrics = self._validate()
            epoch_time = time.time() - t0

            self.train_losses.append(train_metrics["total"])
            self.val_losses.append(val_metrics["total"])
            for key in ["total", "kl", "recon"]:
                self.loss_history[f"train_{key}"].append(train_metrics[key])
                self.loss_history[f"val_{key}"].append(val_metrics[key])
            self.loss_history["beta"].append(self._get_beta())
            self.loss_history["lr"].append(self.optimizer.param_groups[0]["lr"])
            self._log_epoch(train_metrics, val_metrics)
            logger.info(f"  epoch_time={epoch_time:.1f}s")

            self.scheduler.step()

            # Checkpoint
            if self.epoch % self.cfg.checkpoint_every == 0:
                self._save_checkpoint("latest")
            if val_metrics["total"] < self.best_val_loss:
                self.best_val_loss = val_metrics["total"]
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

        # Run evaluation on test set
        from torgen.eval.evaluate import run_evaluation
        eval_dir = os.path.join(self.cfg.checkpoint_dir, "eval")
        results = run_evaluation(
            model=self.model,
            cfg=self.cfg,
            data_dir=self.cfg.local_cache_dir,
            output_dir=eval_dir,
            split="test",
            device=self.device,
        )
        logger.info(f"Evaluation results: {results}")

        if self.wandb_run is not None:
            import wandb
            wandb.finish()
