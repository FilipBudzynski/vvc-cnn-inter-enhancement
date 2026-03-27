import torch
import pytorch_lightning as pl
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities.types import OptimizerLRScheduler
import wandb
from typing import Any, Tuple

from enhancer.models.loss import CharbonnierLoss
from enhancer.config import TrainerConfig


def safe_ms_ssim(pred, target, data_range=1.0, win_size=11):
    """MS-SSIM that handles smaller images gracefully."""
    h, w = pred.shape[2], pred.shape[3]
    min_size = (win_size - 1) * 16
    
    if h <= min_size or w <= min_size:
        return pred.mean() * 0  # Return 0 if too small (will result in loss = 1)
    
    try:
        from enhancer.ssim import ms_ssim
        return ms_ssim(pred, target, data_range=data_range, win_size=win_size)
    except:
        return pred.mean() * 0


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

        self.num_samples: int = getattr(self.config, "num_samples", 4)
        
        self.channel_weights = [0.66666, 0.66666, 0.66666]

        # Initialize metrics once in __init__
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, x: torch.Tensor, metadata: torch.Tensor = None) -> torch.Tensor:
        """Forward pass - handles both legacy (concatenated) and SOTA (separate) formats."""
        output = self.enhancer(x, metadata)
        return output.clamp(0, 1)

    def _split_channels(self, tensor: torch.Tensor):
        return tensor[:, 0:1], tensor[:, 1:2], tensor[:, 2:3]

    def _calculate_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        pY, pU, pV = self._split_channels(pred)
        tY, tU, tV = self._split_channels(target)
        
        weights = torch.tensor(self.channel_weights, device=pred.device)

        mseY = torch.nn.functional.mse_loss(pY, tY)
        mseU = torch.nn.functional.mse_loss(pU, tU)
        mseV = torch.nn.functional.mse_loss(pV, tV)
        mse_loss = (weights * torch.stack([mseY, mseU, mseV])).sum()
        
        l1Y = torch.nn.functional.l1_loss(pY, tY)
        l1U = torch.nn.functional.l1_loss(pU, tU)
        l1V = torch.nn.functional.l1_loss(pV, tV)
        l1_loss = (weights * torch.stack([l1Y, l1U, l1V])).sum()
        
        # Use class-level SSIM metric
        ssim_loss_Y = 1.0 - self.ssim_metric(pY, tY)
        ssim_loss_U = 1.0 - self.ssim_metric(pU, tU)
        ssim_loss_V = 1.0 - self.ssim_metric(pV, tV)
        ssim_loss = (weights * torch.stack([ssim_loss_Y, ssim_loss_U, ssim_loss_V])).sum()
        
        # MS-SSIM
        ms_ssim_loss_Y = 1.0 - safe_ms_ssim(pY, tY, data_range=1.0, win_size=9)
        ms_ssim_loss_U = 1.0 - safe_ms_ssim(pU, tU, data_range=1.0, win_size=9)
        ms_ssim_loss_V = 1.0 - safe_ms_ssim(pV, tV, data_range=1.0, win_size=9)
        ms_ssim_loss = (weights * torch.stack([ms_ssim_loss_Y, ms_ssim_loss_U, ms_ssim_loss_V])).sum()
        
        # Piotr's loss: 0.1*ms_ssim + 0.1*ssim + mse + 0.5*l1
        total_loss = (
            0.1 * ms_ssim_loss + 
            0.1 * ssim_loss + 
            mse_loss + 
            0.5 * l1_loss
        )
        
        return total_loss, (mse_loss, l1_loss, ssim_loss, ms_ssim_loss)

    def training_step(self, batch: Any, _batch_idx: int) -> torch.Tensor:
        # New format: (yuv, original, metadata, info)
        # Legacy format: (x_concat, original, _)
        if len(batch) >= 4:
            yuv, original, metadata, info = batch
            # SOTA mode: YUV + metadata separately
            enhanced = self(yuv, metadata)
        else:
            x, y, _ = batch
            # Legacy mode: concatenated
            enhanced = self(x)
            original = y

        loss, _ = self._calculate_loss(enhanced, original)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        with torch.no_grad():
            batch_psnr = self.psnr_metric(enhanced, original)
            self.log("train_psnr", batch_psnr, prog_bar=True)

        if _batch_idx == 0:
            print(f"\n--- DATA RANGE CHECK ---")
            yuv_input = yuv if len(batch) >= 4 else x[:, :3]
            print(f"Input (VVC)  | Min: {yuv_input.min():.4f} | Max: {yuv_input.max():.4f}")
            print(f"Target (RAW) | Min: {original.min():.4f} | Max: {original.max():.4f}")
            print(f"------------------------\n")

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        if len(batch) >= 4:
            yuv, original, metadata, info = batch
            enhanced = self(yuv, metadata)
        else:
            x, y, _ = batch
            enhanced = self(x)
            original = y

        loss, _ = self._calculate_loss(enhanced, original)

        eY, eU, eV = self._split_channels(enhanced)
        oY, oU, oV = self._split_channels(original)
        
        yuv_input = yuv if len(batch) >= 4 else x[:, :3]
        iY, _, _ = self._split_channels(yuv_input)

        val_psnr_y = self.psnr_metric(eY, oY)
        ref_psnr_y = self.psnr_metric(iY, oY)

        val_ssim_y = self.ssim_metric(eY, oY)
        ref_ssim_y = self.ssim_metric(iY, oY)

        self.log("val_ssim_Y", val_ssim_y, prog_bar=True)
        self.log("val_gain_ssim_Y", val_ssim_y - ref_ssim_y, prog_bar=True)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_psnr_Y", val_psnr_y, prog_bar=True)
        self.log("val_gain_psnr_Y", val_psnr_y - ref_psnr_y, prog_bar=True)

        self.log("val_psnr_U", self.psnr_metric(eU, oU))
        self.log("val_psnr_V", self.psnr_metric(eV, oV))

        if batch_idx == 0:
            self._log_wandb_images(yuv_input, enhanced, original, "val")

        return loss

    def test_step(self, batch: Any, batch_idx: int) -> None:
        if len(batch) >= 4:
            yuv, original, metadata, info = batch
            enhanced = self(yuv, metadata)
            
            # Test with zeroed metadata
            yuv_zeroed = yuv.clone()
            if metadata is not None:
                metadata_zeroed = torch.zeros_like(metadata)
                enhanced_zeroed = self(yuv_zeroed, metadata_zeroed)
            else:
                enhanced_zeroed = enhanced
        else:
            x, orig_chunks, _ = batch
            enhanced = self(x)
            original = orig_chunks
            
            x_zeroed = x.clone()
            x_zeroed[:, 3:] = 0.0
            enhanced_zeroed = self(x_zeroed)

        eY, _, _ = self._split_channels(enhanced)
        eY_zero, _, _ = self._split_channels(enhanced_zeroed)
        oY, _, _ = self._split_channels(original)
        
        yuv_input = yuv if len(batch) >= 4 else x[:, :3]
        iY, _, _ = self._split_channels(yuv_input)

        t_psnr_y = self.psnr_metric(eY, oY)
        z_psnr_y = self.psnr_metric(eY_zero, oY)
        r_psnr_y = self.psnr_metric(iY, oY)

        print(f"\n--- Batch {batch_idx} Results ---")
        print(f"Ref PSNR Y (VVC): {r_psnr_y:.4f}")
        print(f"Model PSNR Y (Real Meta): {t_psnr_y:.4f}")
        print(f"Model PSNR Y (Zero Meta): {z_psnr_y:.4f}")

        self.log_dict(
            {
                "test_psnr_Y": t_psnr_y,
                "test_zero_meta_psnr_Y": z_psnr_y,
                "test_ref_psnr_Y": r_psnr_y,
                "test_gain_Y": t_psnr_y - r_psnr_y,
            }
        )

        if batch_idx == 0:
            self._log_wandb_images(yuv_input, enhanced, original, "test")

    def _yuv_to_rgb(self, yuv: torch.Tensor) -> torch.Tensor:
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
            lq = self._yuv_to_rgb(chunks[i].detach().cpu()).permute(1, 2, 0).numpy()
            pred = self._yuv_to_rgb(enhanced[i].detach().cpu()).permute(1, 2, 0).numpy()
            gt = self._yuv_to_rgb(orig_chunks[i].detach().cpu()).permute(1, 2, 0).numpy()

            img_list.extend(
                [
                    wandb.Image(lq, caption=f"{stage} VVC (RGB)"),
                    wandb.Image(pred, caption=f"{stage} Enhanced (RGB)"),
                    wandb.Image(gt, caption=f"{stage} RAW (RGB)"),
                ]
            )
        self.logger.experiment.log({f"{stage}_previews": img_list})

    def configure_optimizers(self) -> OptimizerLRScheduler:
        weight_decay = self.config.enhancer_lr / 10
        
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.config.enhancer_lr, 
            betas=(0.5, 0.999), 
            weight_decay=weight_decay
        )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[50, 100, 150, 200, 250, 300, 350, 400, 450],
            gamma=0.1
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
