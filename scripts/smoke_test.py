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
            num_queries=8, n_decoder_layers=2,
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
                f"bearing={losses['bearing'].item():.3f}, "
                f"length={losses['length'].item():.3f}, "
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

        # 6. Visual sanity check
        logger.info("Step 6: Plotting outbreak comparison...")
        import os
        import matplotlib.pyplot as plt
        from torgen.viz.plots import plot_outbreak_comparison

        # Always test against the 2011-04-27 Super Outbreak sample
        pt_file = os.path.join(tmp, "2011-04-27.pt")
        out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
        os.makedirs(out_dir, exist_ok=True)
        plot_path = os.path.join(out_dir, "smoke_test_plot.png")
        fig = plot_outbreak_comparison(
            model, pt_file, n_samples=5, threshold=0.5, save_path=plot_path,
        )
        assert os.path.exists(plot_path), "Plot file was not saved"
        logger.info(f"  Plot saved to {plot_path}")
        plt.close(fig)

    logger.info("Smoke test PASSED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"Smoke test FAILED: {e}", exc_info=True)
        sys.exit(1)
