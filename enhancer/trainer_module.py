import torch
import wandb
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional.classification import accuracy
from .ssim import SSIM, MS_SSIM
from .config import TrainerConfig, TrainingMode
from .dataset import VVCDataset, FrameDataset


class TrainerModule(pl.LightningModule):
    def __init__(
        self,
        config: TrainerConfig,
        enhancer,
        discriminator,
        test_full_frames: bool = False,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()

        self.enhancer = enhancer
        self.discriminator = discriminator
        self.mode = config.mode
        self.config = config.current

        self.separation_epochs = config.separation_epochs

        self.enhancer_lr = self.config.enhancer_lr
        self.discriminator_lr = self.config.discriminator_lr

        self.betas = self.config.betas
        self.momentum = self.config.momentum
        self.discriminator_min_loss = self.config.discriminator_min_loss
        self.enhancer_min_loss = self.config.enhancer_min_loss

        self.enhancer_losses = [1.0]
        self.discriminator_losses = [1.0]
        self.probe = self.config.probe

        self.num_samples = self.config.num_samples
        self.ssim = SSIM(data_range=1.0, win_size=9, per_channel=True)
        self.msssim = MS_SSIM(data_range=1.0, win_size=9, per_channel=True)

        self.test_full_frames = test_full_frames
        self.channels_grad_scales = config.channels_grad_scales

    def forward(self, chunks, metadata, vtm_features=None):
        return self.enhancer(chunks, metadata, vtm_features)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def what_to_train(self):
        if self.mode == TrainingMode.ENHANCER:
            return True, None
        elif self.mode == TrainingMode.DISCRIMINATOR:
            return None, True
        else:
            e_loss = np.mean(self.enhancer_losses)
            d_loss = np.mean(self.discriminator_losses)

            e_train = e_loss >= self.enhancer_min_loss
            d_train = d_loss >= self.discriminator_min_loss

            if not e_train and not d_train:
                return True, True

            return bool(e_train), bool(d_train)

        return True, True

    def g_step(self, chunks, orig_chunks, metadata, vtm_features=None, stage="train"):
        enhanced = self(chunks, metadata, vtm_features)

        prefix = "" if stage == "train" else stage + "_"

        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(chunks.size(0), 1)
        valid = valid.type_as(chunks)

        # split channels
        def split(x):
            return x[:, [0]], x[:, [1]], x[:, [2]]

        eY, eU, eV = split(enhanced)
        oY, oU, oV = split(orig_chunks)

        def to_tensor(x):
            return torch.Tensor(x).to(enhanced.device)

        channels_grad_scales = to_tensor(self.channels_grad_scales)

        # those are by channel
        ssim_l = 1 - self.ssim(orig_chunks, enhanced)
        ssimY, ssimU, ssimV = ssim_l
        ssim_loss = (channels_grad_scales * ssim_l).sum()

        self.log(f"{prefix}g_ssimY_loss", ssimY, prog_bar=False)
        self.log(f"{prefix}g_ssimU_loss", ssimU, prog_bar=False)
        self.log(f"{prefix}g_ssimV_loss", ssimV, prog_bar=False)
        self.log(f"{prefix}g_ssim_loss", ssim_loss, prog_bar=False)
        self.log(f"{prefix}g_ssim", ssim_l.mean(), prog_bar=False)

        msssim_l = 1 - self.msssim(orig_chunks, enhanced)
        msssimY, msssimU, msssimV = msssim_l
        msssim_loss = (channels_grad_scales * msssim_l).sum()

        self.log(f"{prefix}g_msssimY_loss", msssimY, prog_bar=False)
        self.log(f"{prefix}g_msssimU_loss", msssimU, prog_bar=False)
        self.log(f"{prefix}g_msssimV_loss", msssimV, prog_bar=False)
        self.log(f"{prefix}g_msssim_loss", msssim_loss, prog_bar=False)
        self.log(f"{prefix}g_msssim", msssim_l.mean(), prog_bar=False)

        mseY = F.mse_loss(oY, eY)
        mseU = F.mse_loss(oU, eU)
        mseV = F.mse_loss(oV, eV)
        mse_loss = (channels_grad_scales * to_tensor([mseY, mseU, mseV])).sum()
        self.log(f"{prefix}g_mse_loss", mse_loss, prog_bar=False)
        self.log(f"{prefix}g_mseY_loss", mseY, prog_bar=False)
        self.log(f"{prefix}g_mseU_loss", mseU, prog_bar=False)
        self.log(f"{prefix}g_mseV_loss", mseV, prog_bar=False)

        l1Y = F.l1_loss(oY, eY)
        l1U = F.l1_loss(oU, eU)
        l1V = F.l1_loss(oV, eV)
        l1_loss = (channels_grad_scales * to_tensor([l1Y, l1U, l1V])).sum()
        self.log(f"{prefix}g_l1_loss", l1_loss, prog_bar=False)
        self.log(f"{prefix}g_l1Y_loss", l1Y, prog_bar=False)
        self.log(f"{prefix}g_l1U_loss", l1U, prog_bar=False)
        self.log(f"{prefix}g_l1V_loss", l1V, prog_bar=False)

        if (
            self.mode == TrainingMode.GAN
            and self.current_epoch > self.separation_epochs
        ):
            preds = self.discriminator(enhanced)
            gd_loss = self.adversarial_loss(preds, valid)
            g_loss = (
                0.1 * msssim_loss
                + 0.1 * ssim_loss
                + mse_loss
                + 0.5 * l1_loss
                + 0.1 * gd_loss
            )
            self.log(f"{prefix}g_d_loss", gd_loss, prog_bar=True)
            self.enhancer_losses.append(gd_loss.item())
            self.enhancer_losses = self.enhancer_losses[: self.probe]
        else:
            preds = None
            g_loss = 0.1 * msssim_loss + 0.1 * ssim_loss + mse_loss + 0.5 * l1_loss

        self.log(f"{prefix}g_loss", g_loss, prog_bar=True)

        if stage != "train":
            cY, cU, cV = split(chunks)

            enhanced_psnr = psnr(
                enhanced,
                orig_chunks,
            )
            orig_psnr = psnr(
                chunks,
                orig_chunks,
            )

            epsnrY = psnr(eY, oY)
            epsnrU = psnr(eU, oU)
            epsnrV = psnr(eV, oV)

            cpsnrY = psnr(cY, oY)
            cpsnrU = psnr(cU, oU)
            cpsnrV = psnr(cV, oV)

            enhanced_ssim = ssim(
                enhanced,
                orig_chunks,
            )
            orig_ssim = ssim(
                chunks,
                orig_chunks,
            )
            self.log_dict(
                {
                    f"{prefix}psnr": enhanced_psnr,
                    f"{prefix}psnrY": epsnrY,
                    f"{prefix}psnrU": epsnrU,
                    f"{prefix}psnrV": epsnrV,
                    f"{prefix}ref_psnr": orig_psnr,
                    f"{prefix}ref_psnrY": cpsnrY,
                    f"{prefix}ref_psnrU": cpsnrU,
                    f"{prefix}ref_psnrV": cpsnrV,
                    f"{prefix}ssim": enhanced_ssim,
                    f"{prefix}ref_ssim": orig_ssim,
                },
            )

        return enhanced, preds, g_loss

    def d_step(self, fake_chunks, orig_chunks, stage="train"):
        prefix = "" if stage == "train" else stage + "_"

        valid = torch.ones(orig_chunks.size(0), 1)
        valid = valid.type_as(orig_chunks)
        real_pred = self.discriminator(orig_chunks)
        real_loss = self.adversarial_loss(real_pred, valid)
        real_accuracy = accuracy(real_pred, valid, task="binary")

        # how well can it label as fake?
        fake = torch.zeros(orig_chunks.size(0), 1)
        fake = fake.type_as(orig_chunks)

        fake_preds = self.discriminator(fake_chunks)
        fake_loss = self.adversarial_loss(fake_preds, fake)
        fake_accuracy = accuracy(fake_preds, fake, task="binary")

        acc = (real_accuracy + fake_accuracy) / 2

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2

        self.discriminator_losses.append(d_loss.item())
        self.discriminator_losses = self.discriminator_losses[: self.probe]

        self.log(f"{prefix}d_loss", d_loss, prog_bar=True)
        self.log(f"{prefix}d_real_loss", real_loss, prog_bar=False)
        self.log(f"{prefix}d_fake_loss", fake_loss, prog_bar=False)
        self.log(f"{prefix}d_real_acc", real_accuracy, prog_bar=False)
        self.log(f"{prefix}d_fake_acc", fake_accuracy, prog_bar=False)
        self.log(f"{prefix}d_acc", acc, prog_bar=True)

        return fake_preds, real_pred, d_loss

    def log_images(
        self, enhanced, chunks, orig_chunks, preds, real_preds, stage="train"
    ):
        prefix = "" if stage == "train" else stage + "_"

        log = {"uncompressed": [], "decompressed": []}

        if self.mode != TrainingMode.DISCRIMINATOR:
            log["enhanced"] = []

        for i in range(self.num_samples):
            orig = orig_chunks[i].cpu()
            dec = chunks[i].cpu()

            if self.mode != TrainingMode.DISCRIMINATOR:
                enh = enhanced[i].cpu()
                log["enhanced"].append(
                    wandb.Image(
                        enh,
                        caption=f"Pred: {preds[i].item()}"
                        if self.mode == TrainingMode.GAN
                        else f"ENH: {i}",
                    )
                )

            log["uncompressed"].append(
                wandb.Image(
                    orig,
                    caption=f"Pred: {real_preds[i].item()}"
                    if self.mode != TrainingMode.ENHANCER
                    else f"UNC: {i}",
                )
            )

            log["decompressed"].append(
                wandb.Image(
                    dec,
                    caption=f"Pred: {preds[i].item()}"
                    if self.mode == TrainingMode.DISCRIMINATOR
                    else f"DEC: {i}",
                )
            )

        log = {prefix + key: value for key, value in log.items()}
        self.logger.experiment.log(log)

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()
        chunks, orig_chunks, metadata, _, vtm_features = batch
        e_train, d_train = self.what_to_train()
        preds = None

        # train ENHANCE!
        # ENHANCE!
        if e_train:
            # with gradient
            g_opt.zero_grad()

            enhanced, preds, g_loss = self.g_step(chunks, orig_chunks, metadata, vtm_features)
            fake_chunks = enhanced.detach()

            self.manual_backward(g_loss)
            g_opt.step()
        elif e_train is not None:
            # just log enhancer loss
            enhanced, preds, g_loss = self.g_step(chunks, orig_chunks, metadata, vtm_features)
            fake_chunks = enhanced.detach()
        else:
            fake_chunks = chunks
            enhanced = None

        # train discriminator
        if d_train:
            d_opt.zero_grad()

            fake_preds, real_preds, d_loss = self.d_step(fake_chunks, orig_chunks)

            self.manual_backward(d_loss)
            d_opt.step()
        elif d_train is not None:
            fake_preds, real_preds, d_loss = self.d_step(fake_chunks, orig_chunks)
        else:
            real_preds = None
            fake_preds = preds

        if batch_idx % 100 == 0:
            self.log_images(
                enhanced,
                chunks,
                orig_chunks,
                fake_preds,
                real_preds,
            )

    def on_train_epoch_end(self):
        schs = self.lr_schedulers()

        if schs is None:
            return

        if not isinstance(schs, (tuple, list)):
            schs = [schs]

        for sch in schs:
            sch.step()

    def validation_step(self, batch, batch_idx):
        chunks, orig_chunks, metadata, _, vtm_features = batch
        preds = None

        # train ENHANCE!
        # ENHANCE!
        if self.mode != TrainingMode.DISCRIMINATOR:
            # with gradient
            enhanced, preds, g_loss = self.g_step(chunks, orig_chunks, metadata, vtm_features, "val")
            fake_chunks = enhanced.detach()
        else:
            fake_chunks = chunks
            enhanced = None

        # train discriminator
        if self.mode != TrainingMode.ENHANCER:
            fake_preds, real_preds, d_loss = self.d_step(
                fake_chunks, orig_chunks, "val"
            )
        else:
            real_preds = None
            fake_preds = preds

        if batch_idx % 100 == 0:
            self.log_images(
                enhanced,
                chunks,
                orig_chunks,
                fake_preds,
                real_preds,
                "val",
            )

    def test_step(self, batch, batch_idx):
        chunks, orig_chunks, metadata, _, vtm_features = batch
        preds = None

        # train ENHANCE!
        # ENHANCE!
        if self.mode != TrainingMode.DISCRIMINATOR:
            # with gradient
            enhanced, preds, g_loss = self.g_step(chunks, orig_chunks, metadata, vtm_features, "test")
            fake_chunks = enhanced.detach()
        else:
            fake_chunks = chunks
            enhanced = None

        # train discriminator
        if self.mode != TrainingMode.ENHANCER:
            fake_preds, real_preds, d_loss = self.d_step(
                fake_chunks, orig_chunks, "test"
            )
        else:
            real_preds = None
            fake_preds = preds

        if batch_idx % 10 == 0:
            self.log_images(
                enhanced,
                chunks,
                orig_chunks,
                fake_preds,
                real_preds,
                "test",
            )

    def predict_step(self, batch, batch_idx):
        chunks, _, metadata, chunk_objs = batch

        # ENHANCE!
        enhanced = self(chunks, metadata)

        for i, chunk_data in enumerate(enhanced):
            chunk = [c[i].cpu() if hasattr(c[i], "cpu") else c[i] for c in chunk_objs]
            if self.test_full_frames:
                FrameDataset.save_frame(
                    chunk, chunk_data.cpu().numpy(), self.config.saved_chunk_folder
                )
            else:
                VVCDataset.save_chunk(
                    chunk, chunk_data.cpu().numpy(), self.config.saved_chunk_folder
                )

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.enhancer.parameters(),
            lr=self.enhancer_lr,
            betas=self.betas,
            weight_decay=self.enhancer_lr / 10,
        )
        opt_d = torch.optim.SGD(
            self.discriminator.parameters(),
            lr=self.discriminator_lr,
            momentum=self.momentum,
            weight_decay=self.discriminator_lr / 10,
        )

        lr_schedulers = []

        if self.config.enhancer_scheduler is True:
            lr_schedulers.append(
                {
                    "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                        opt_g,
                        milestones=self.config.enhancer_scheduler_milestones,
                        gamma=self.config.enhancer_scheduler_gamma,
                    ),
                    "interval": "epoch",
                    "frequency": 1,
                }
            )

        if self.config.discriminator_scheduler is True:
            lr_schedulers.append(
                {
                    "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                        opt_d,
                        milestones=self.config.discriminator_scheduler_milestones,
                        gamma=self.config.discriminator_scheduler_gamma,
                    ),
                    "interval": "epoch",
                    "frequency": 1,
                }
            )

        return [opt_g, opt_d], lr_schedulers