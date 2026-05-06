"""
Snow-Wide retraining on the existing 9-channel precomputed martell dataset.

The original snow-wide-gradient checkpoint expected 19-channel metadata, but
no precompute script in any branch produces 19 channels — that recipe is
unrecoverable. This script retrains Snow-Wide with metadata_channels=9 (same
as Martell) so it can be fairly evaluated against Martell and VVC-PPFF.

Loss matches the original snow-wide-gradient training:
    0.5*L1 + 0.15*MS-SSIM + 0.2*GradLoss + 0.15*Laplacian
Schedule: Adam(lr=1e-4, wd=1e-4), MultiStepLR milestones=[50, 100, 150]
Data: data/precomputed_martell (the only 9-ch precomputed set we have).
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio

from enhancer.dataset_blackfyre import BlackfyreDataset
from enhancer.models.snow_wide import SnowWideEnhancer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_BATCH = 8
DEFAULT_EPOCHS = 200
DEFAULT_LR = 1e-4
DEFAULT_PATCH = 132
DEFAULT_DATA = "data/precomputed_martell"
DEFAULT_CKPT_PREFIX = "snow_wide_9ch"


def compute_loss(enhanced, original):
    l1 = F.l1_loss(enhanced, original)

    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=enhanced.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                      device=enhanced.device).view(1, 1, 3, 3)

    def grad_xy(img):
        flat = img.reshape(-1, 1, img.shape[2], img.shape[3])
        gx = F.conv2d(flat, kx, padding=1).view_as(img)
        gy = F.conv2d(flat, ky, padding=1).view_as(img)
        return gx, gy

    g_e_x, g_e_y = grad_xy(enhanced)
    g_o_x, g_o_y = grad_xy(original)
    grad_loss = F.l1_loss(g_e_x, g_o_x) + F.l1_loss(g_e_y, g_o_y)

    lap_kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                              device=enhanced.device).view(1, 1, 3, 3)
    lap_e = F.conv2d(enhanced.reshape(-1, 1, enhanced.shape[2], enhanced.shape[3]),
                     lap_kernel, padding=1).view_as(enhanced)
    lap_o = F.conv2d(original.reshape(-1, 1, original.shape[2], original.shape[3]),
                     lap_kernel, padding=1).view_as(original)
    lap_loss = F.l1_loss(lap_e, lap_o)

    ms = ms_ssim(enhanced, original, data_range=1.0, size_average=True, win_size=7)
    ms_ssim_loss = 1 - ms

    total = 0.5 * l1 + 0.15 * ms_ssim_loss + 0.2 * grad_loss + 0.15 * lap_loss
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--patch-size", type=int, default=DEFAULT_PATCH)
    p.add_argument("--data-dir", default=DEFAULT_DATA)
    p.add_argument("--ckpt-prefix", default=DEFAULT_CKPT_PREFIX)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-every", type=int, default=20)
    args = p.parse_args()

    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    import random; random.seed(42); np.random.seed(42)

    class Cfg:
        metadata_channels = 9
        base_channels = 64

    train_ds = BlackfyreDataset(args.data_dir, patch_size=args.patch_size, split="train")
    val_ds = BlackfyreDataset(args.data_dir, patch_size=args.patch_size, split="val")
    print(f"Train samples: {len(train_ds)}  Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = SnowWideEnhancer(Cfg()).to(DEVICE)
    print(f"Model params: {sum(x.numel() for x in model.parameters()):,}")

    optim_ = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.MultiStepLR(optim_, milestones=[50, 100, 150])
    psnr_e = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    psnr_i = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)

    os.makedirs("checkpoints", exist_ok=True)
    log_path = f"snow_wide_9ch_train.log"
    print(f"Logging to {log_path}")

    best_val_gain = -1e9
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for batch_idx, (decoded, original, metadata, _info) in enumerate(train_loader):
            prev_frame, curr_frame, next_frame = decoded
            prev_frame = prev_frame.to(DEVICE, non_blocking=True)
            curr_frame = curr_frame.to(DEVICE, non_blocking=True)
            next_frame = next_frame.to(DEVICE, non_blocking=True)
            original = original.to(DEVICE, non_blocking=True)
            metadata = metadata.to(DEVICE, non_blocking=True)

            optim_.zero_grad()
            enhanced = model(curr_frame, prev_frame, next_frame, metadata).clamp(0, 1)
            loss = compute_loss(enhanced, original)
            loss.backward()
            optim_.step()
            train_loss += float(loss.detach())

        sched.step()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_gain = 0.0; val_in = 0.0; val_enh = 0.0; n = 0
        with torch.no_grad():
            for decoded, original, metadata, _info in val_loader:
                prev_frame, curr_frame, next_frame = decoded
                prev_frame = prev_frame.to(DEVICE, non_blocking=True)
                curr_frame = curr_frame.to(DEVICE, non_blocking=True)
                next_frame = next_frame.to(DEVICE, non_blocking=True)
                original = original.to(DEVICE, non_blocking=True)
                metadata = metadata.to(DEVICE, non_blocking=True)
                enhanced = model(curr_frame, prev_frame, next_frame, metadata).clamp(0, 1)
                for i in range(enhanced.shape[0]):
                    pe = psnr_e(enhanced[i:i+1], original[i:i+1]).item()
                    pi = psnr_i(curr_frame[i:i+1], original[i:i+1]).item()
                    val_enh += pe; val_in += pi; val_gain += (pe - pi); n += 1
                    psnr_e.reset(); psnr_i.reset()

        avg_in = val_in / n; avg_enh = val_enh / n; avg_gain = val_gain / n
        dt = time.time() - t0
        line = (f"epoch {epoch:3d}  loss={train_loss:.4f}  "
                f"val_in={avg_in:.3f}  val_enh={avg_enh:.3f}  gain={avg_gain:+.4f}  "
                f"lr={optim_.param_groups[0]['lr']:.2e}  ({dt:.0f}s)")
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt = f"checkpoints/{args.ckpt_prefix}_epoch_{epoch + 1}.pt"
            torch.save(model.state_dict(), ckpt)
            print(f"  saved {ckpt}", flush=True)
        if avg_gain > best_val_gain:
            best_val_gain = avg_gain
            best_path = f"checkpoints/{args.ckpt_prefix}_best.pt"
            torch.save(model.state_dict(), best_path)


if __name__ == "__main__":
    main()
