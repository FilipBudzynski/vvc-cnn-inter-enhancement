import torch
import pytorch_lightning as pl
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities.types import OptimizerLRScheduler
import wandb
from typing import Any, cast

# Absolute imports
from enhancer.models.loss import CharbonnierLoss
from enhancer.config import TrainerConfig


class TrainerModule(pl.LightningModule):
    def __init__(
        self,
        config: TrainerConfig,
        enhancer: torch.nn.Module,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["enhancer"])

        self.enhancer = enhancer
        self.config = config.current
        self.criterion = CharbonnierLoss()

        # Pull num_samples safely from config
        self.num_samples: int = getattr(self.config, "num_samples", 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x is a single tensor of shape [Batch, 10, H, W]
        (3 YUV channels + 7 metadata channels)
        """
        return self.enhancer(x)

    def _split_channels(self, tensor: torch.Tensor):
        """Splits a YUV tensor [B, 3, H, W] into individual Y, U, and V components."""
        return tensor[:, 0:1], tensor[:, 1:2], tensor[:, 2:3]

    def _calculate_weighted_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Calculates Charbonnier loss with channel weighting."""
        pY, pU, pV = self._split_channels(pred)
        tY, tU, tV = self._split_channels(target)

        loss_Y = self.criterion(pY, tY)
        loss_U = self.criterion(pU, tU)
        loss_V = self.criterion(pV, tV)

        total_loss = (1.0 * loss_Y) + (0.5 * loss_U) + (0.5 * loss_V)
        return total_loss, (loss_Y, loss_U, loss_V)

    def training_step(self, batch: Any, _batch_idx: int) -> torch.Tensor:
        x, y, _ = batch
        # --- DIAGNOSTIC PRINTS ---
        if _batch_idx == 0:
            print(f"\n--- DATA RANGE CHECK ---")
            print(
                f"Input (VVC)  | Min: {x[:, :3].min():.4f} | Max: {x[:, :3].max():.4f}"
            )
            print(f"Target (RAW) | Min: {y.min():.4f} | Max: {y.max():.4f}")
            print(
                f"Metadata     | Min: {x[:, 3:].min():.4f} | Max: {x[:, 3:].max():.4f}"
            )
            print(f"------------------------\n")
        # -------------------------

        enhanced = self(x)
        loss, _ = self._calculate_weighted_loss(enhanced, y)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        with torch.no_grad():
            batch_psnr = psnr(enhanced, y, data_range=1.0)
            self.log("train_psnr", batch_psnr, prog_bar=True)

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y, _ = batch
        enhanced = self(x)
        loss, _ = self._calculate_weighted_loss(enhanced, y)

        # Piotr style: split and calculate on 0-1 range
        eY, eU, eV = self._split_channels(enhanced)
        oY, oU, oV = self._split_channels(y)
        iY, _, _ = self._split_channels(x[:, :3])

        # PSNR on Y channel (Data range 1.0)
        val_psnr_y = cast(torch.Tensor, psnr(eY, oY, data_range=1.0))
        ref_psnr_y = cast(torch.Tensor, psnr(iY, oY, data_range=1.0))

        # SSIM on Y channel
        val_ssim_y = cast(torch.Tensor, ssim(eY, oY, data_range=1.0))
        ref_ssim_y = cast(torch.Tensor, ssim(iY, oY, data_range=1.0))

        self.log("val_ssim_Y", val_ssim_y, prog_bar=True)
        self.log("val_gain_ssim_Y", val_ssim_y - ref_ssim_y, prog_bar=True)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_psnr_Y", val_psnr_y, prog_bar=True)
        self.log("val_gain_psnr_Y", val_psnr_y - ref_psnr_y, prog_bar=True)

        # Piotr also logs U and V PSNR specifically
        self.log("val_psnr_U", psnr(eU, oU, data_range=1.0))
        self.log("val_psnr_V", psnr(eV, oV, data_range=1.0))

        if batch_idx == 0:
            self._log_wandb_images(x[:, :3], enhanced, y, "val")

        return loss

    def test_step(self, batch: Any, batch_idx: int) -> None:
        x, orig_chunks, _ = batch

        # 1. Standard Inference (using real metadata)
        enhanced = self(x)

        # 2. Zero-Metadata Diagnostic (Hacking metadata to zero)
        x_zeroed = x.clone()
        x_zeroed[:, 3:] = 0.0
        enhanced_zeroed = self(x_zeroed)

        # Split channels for metrics
        eY, _, _ = self._split_channels(enhanced)
        eY_zero, _, _ = self._split_channels(enhanced_zeroed)
        oY, _, _ = self._split_channels(orig_chunks)
        iY, _, _ = self._split_channels(x[:, :3])

        # Metrics
        t_psnr_y = psnr(eY * 255.0, oY * 255.0, data_range=255.0)
        z_psnr_y = psnr(eY_zero * 255.0, oY * 255.0, data_range=255.0)
        r_psnr_y = psnr(iY * 255.0, oY * 255.0, data_range=255.0)

        # Print to terminal for immediate feedback
        print(f"\n--- Batch {batch_idx} Results ---")
        print(f"Ref PSNR Y (VVC): {r_psnr_y:.4f}")
        print(f"Model PSNR Y (Real Meta): {t_psnr_y:.4f}")
        print(f"Model PSNR Y (Zero Meta): {z_psnr_y:.4f}")
        print(f"Mean VVC-vs-RAW Diff: {(x[:, :3] - orig_chunks).abs().mean():.6f}")

        self.log_dict(
            {
                "test_psnr_Y": t_psnr_y,
                "test_zero_meta_psnr_Y": z_psnr_y,
                "test_ref_psnr_Y": r_psnr_y,
                "test_gain_Y": t_psnr_y - r_psnr_y,
            }
        )

        if batch_idx == 0:
            self._log_wandb_images(x[:, :3], enhanced, orig_chunks, "test")

    def _yuv_to_rgb(self, yuv: torch.Tensor) -> torch.Tensor:
        """Converts YUV [3, H, W] to RGB [3, H, W] for WandB."""
        y, u, v = yuv[0], yuv[1], yuv[2]
        r = y + 1.402 * (v - 0.5)
        g = y - 0.344136 * (u - 0.5) - 0.714136 * (v - 0.5)
        b = y + 1.772 * (u - 0.5)
        return torch.stack([r, g, b]).clamp(0, 1)

    def _log_wandb_images(self, chunks, enhanced, orig_chunks, stage):
        if not isinstance(self.logger, WandbLogger):
            return
        count = min(chunks.shape[0], self.num_samples)
        img_list = []

        for i in range(count):
            # Convert to RGB before sending to WandB
            lq = self._yuv_to_rgb(chunks[i].detach().cpu()).permute(1, 2, 0).numpy()
            pred = self._yuv_to_rgb(enhanced[i].detach().cpu()).permute(1, 2, 0).numpy()
            gt = (
                self._yuv_to_rgb(orig_chunks[i].detach().cpu()).permute(1, 2, 0).numpy()
            )

            img_list.extend(
                [
                    wandb.Image(lq, caption=f"{stage} VVC (RGB)"),
                    wandb.Image(pred, caption=f"{stage} Enhanced (RGB)"),
                    wandb.Image(gt, caption=f"{stage} RAW (RGB)"),
                ]
            )
        self.logger.experiment.log({f"{stage}_previews": img_list})

    def configure_optimizers(self) -> OptimizerLRScheduler:
        max_epochs = self.config.epochs
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config.enhancer_lr, weight_decay=1e-4
        )

        # 1. Linear Warmup: Start at 10% of LR and reach 100% in 5 epochs
        warmup_steps = 5
        scheduler1 = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
        )

        # 2. Cosine Decay: After warmup, slowly decay the LR in a curve
        # This is better than Plateau because it keeps the model "moving"
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs - warmup_steps
        )

        # Combine them: scheduler1 for 5 epochs, then scheduler2
        combined_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": combined_scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
