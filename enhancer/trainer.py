import torch
import pytorch_lightning as pl
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities.types import OptimizerLRScheduler
import wandb
from typing import Any

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

    def forward(self, chunks: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        return self.enhancer(chunks, metadata)

    def training_step(self, batch: Any, _batch_idx: int) -> torch.Tensor:
        # x: [B, 10, 128, 128] (Y, U, V, 7 metadata)
        # y: [B, 3, 128, 128] (origina lY, U, V)
        x, y, _ = batch

        enhanced = self.enhancer(x)
        loss = self.criterion(enhanced, y)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        with torch.no_grad():
            batch_psnr = psnr(enhanced, y, data_range=1.0)
            self.log("train_psnr", batch_psnr, prog_bar=True)

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        # chunks, orig_chunks, metadata, _ = batch
        x, y, _ = batch

        enhanced = self.enhancer(x)
        val_loss = self.criterion(enhanced, y)

        # Calculate metrics
        v_psnr = psnr(enhanced, y, data_range=1.0)
        v_ssim = ssim(enhanced, y, data_range=1.0)
        # Use only first 3 channels (YUV) for reference PSNR
        r_psnr = psnr(x[:, :3, :, :], y, data_range=1.0)

        # Logging individual metrics to avoid Dict typing issues
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_psnr", v_psnr, prog_bar=True)
        self.log("val_ssim", v_ssim, prog_bar=True)
        self.log("val_psnr_gain", v_psnr - r_psnr, prog_bar=True)

        if batch_idx == 0:
            self._log_wandb_images(x[:, :3, :, :], enhanced, y, "val")

        return val_loss

    def test_step(self, batch: Any, batch_idx: int) -> None:
        chunks, orig_chunks, metadata, _ = batch
        enhanced = self(chunks, metadata)

        t_psnr = psnr(enhanced, orig_chunks, data_range=1.0)
        self.log("test_psnr", t_psnr)

        if batch_idx == 0:
            self._log_wandb_images(chunks, enhanced, orig_chunks, "test")

    def _log_wandb_images(
        self,
        chunks: torch.Tensor,
        enhanced: torch.Tensor,
        orig_chunks: torch.Tensor,
        stage: str,
    ):
        if not isinstance(self.logger, WandbLogger):
            return

        count = min(chunks.shape[0], self.num_samples)
        img_list = []

        for i in range(count):
            # lq = chunks[i, 0].detach().cpu().numpy()
            # pred = enhanced[i, 0].detach().cpu().numpy()
            # gt = orig_chunks[i, 0].detach().cpu().numpy()
            lq = chunks[i, :3].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
            pred = enhanced[i].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
            gt = orig_chunks[i].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)

            img_list.extend(
                [
                    wandb.Image(lq, caption=f"{stage} VVC"),
                    wandb.Image(pred, caption=f"{stage} Enhanced"),
                    wandb.Image(gt, caption=f"{stage} RAW"),
                ]
            )

        self.logger.experiment.log({f"{stage}_previews": img_list})

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config.enhancer_lr, weight_decay=1e-4
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # Returning the specific dictionary format Lightning expects
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
