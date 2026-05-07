"""
Martell hybrid-loss retraining: combine the strong Y-focused loss from
the original Martell training (which got Y BD-rate -4.45 %) with an
explicit chroma MSE term, so chroma gets direct gradient instead of
being a side-effect of the L1 component.

Total loss:
    L_y     = 0.5*L1 + 0.15*MS-SSIM + 0.2*GradLoss + 0.15*Laplacian   (on Y plane)
    L_uv    = MSE(U_pred, U_orig) + MSE(V_pred, V_orig)                (chroma only)
    total   = L_y + chroma_weight * L_uv

`chroma_weight` is the only knob; default 1.0 (chroma roughly as much
gradient as Y per pixel, since L_uv is averaged over twice as many
pixels as a single Y plane).
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

from enhancer.dataset_blackfyre import BlackfyreDataset
from enhancer.models.snow_wide import SnowWideEnhancer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def y_loss(enhanced_y: torch.Tensor, original_y: torch.Tensor) -> torch.Tensor:
    """Original Martell multi-term loss restricted to a single Y plane.

    enhanced_y and original_y are both [B, 1, H, W]."""
    l1 = F.l1_loss(enhanced_y, original_y)

    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=enhanced_y.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                      device=enhanced_y.device).view(1, 1, 3, 3)
    g_e_x = F.conv2d(enhanced_y, kx, padding=1)
    g_e_y = F.conv2d(enhanced_y, ky, padding=1)
    g_o_x = F.conv2d(original_y, kx, padding=1)
    g_o_y = F.conv2d(original_y, ky, padding=1)
    grad_loss = F.l1_loss(g_e_x, g_o_x) + F.l1_loss(g_e_y, g_o_y)

    lap_kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                              device=enhanced_y.device).view(1, 1, 3, 3)
    lap_e = F.conv2d(enhanced_y, lap_kernel, padding=1)
    lap_o = F.conv2d(original_y, lap_kernel, padding=1)
    lap_loss = F.l1_loss(lap_e, lap_o)

    # MS-SSIM expects 3-channel for default win_size; replicate Y to 3 channels
    # so we can reuse pytorch_msssim. Result is the same as 1-channel MS-SSIM
    # because each channel is identical.
    ey3 = enhanced_y.expand(-1, 3, -1, -1)
    oy3 = original_y.expand(-1, 3, -1, -1)
    ms = ms_ssim(ey3, oy3, data_range=1.0, size_average=True, win_size=7)
    ms_ssim_loss = 1 - ms

    return 0.5 * l1 + 0.15 * ms_ssim_loss + 0.2 * grad_loss + 0.15 * lap_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patch-size", type=int, default=132)
    p.add_argument("--data-dir", default="data/precomputed_martell")
    p.add_argument("--ckpt-prefix", default="martell_hybrid")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--chroma-weight", type=float, default=1.0,
                   help="multiplier on the explicit chroma MSE term")
    p.add_argument("--resume", type=str, default=None,
                   help="path to checkpoint to fine-tune from")
    args = p.parse_args()

    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    import random; random.seed(42); np.random.seed(42)

    class Cfg:
        metadata_channels = 9
        base_channels = 64

    train_ds = BlackfyreDataset(args.data_dir, patch_size=args.patch_size, split="train")
    val_ds = BlackfyreDataset(args.data_dir, patch_size=args.patch_size, split="val")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = SnowWideEnhancer(Cfg()).to(DEVICE)
    if args.resume:
        state = torch.load(args.resume, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        print(f"Resumed from {args.resume}")
    print(f"Params: {sum(x.numel() for x in model.parameters()):,}")
    print(f"Loss = Y_multi_term + {args.chroma_weight} * MSE(chroma)")

    optim_ = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.MultiStepLR(optim_, milestones=[50, 100, 150, 200, 300])

    os.makedirs("checkpoints", exist_ok=True)
    log_path = f"{args.ckpt_prefix}_train.log"
    print(f"Logging to {log_path}")

    best_val_gain = -1e9
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        sum_loss = sum_loss_y = sum_loss_uv = 0.0

        for decoded, original, metadata, _info in train_loader:
            prev_frame, curr_frame, next_frame = decoded
            prev_frame = prev_frame.to(DEVICE, non_blocking=True)
            curr_frame = curr_frame.to(DEVICE, non_blocking=True)
            next_frame = next_frame.to(DEVICE, non_blocking=True)
            original = original.to(DEVICE, non_blocking=True)
            metadata = metadata.to(DEVICE, non_blocking=True)

            optim_.zero_grad()
            enhanced = model(curr_frame, prev_frame, next_frame, metadata).clamp(0, 1)
            loss_y = y_loss(enhanced[:, 0:1], original[:, 0:1])
            loss_uv = F.mse_loss(enhanced[:, 1:3], original[:, 1:3])
            loss = loss_y + args.chroma_weight * loss_uv

            loss.backward()
            optim_.step()
            sum_loss += float(loss.detach())
            sum_loss_y += float(loss_y.detach())
            sum_loss_uv += float(loss_uv.detach())

        sched.step()
        n_batches = len(train_loader)

        # Validation: per-channel PSNR gain
        model.eval()
        val_in = val_enh = 0.0; val_gy = val_gu = val_gv = 0.0; n = 0
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
                    for ch_idx in range(3):
                        mse_e = F.mse_loss(enhanced[i:i+1, ch_idx:ch_idx+1],
                                           original[i:i+1, ch_idx:ch_idx+1]).item()
                        mse_in = F.mse_loss(curr_frame[i:i+1, ch_idx:ch_idx+1],
                                            original[i:i+1, ch_idx:ch_idx+1]).item()
                        d = (10 * np.log10(1.0 / max(mse_e, 1e-12))
                             - 10 * np.log10(1.0 / max(mse_in, 1e-12)))
                        if ch_idx == 0: val_gy += d
                        elif ch_idx == 1: val_gu += d
                        else: val_gv += d
                    val_enh += 10 * np.log10(1.0 / max(F.mse_loss(enhanced[i:i+1], original[i:i+1]).item(), 1e-12))
                    val_in += 10 * np.log10(1.0 / max(F.mse_loss(curr_frame[i:i+1], original[i:i+1]).item(), 1e-12))
                    n += 1

        avg_gy = val_gy / n; avg_gu = val_gu / n; avg_gv = val_gv / n
        avg_total_gain = avg_gy + avg_gu + avg_gv
        dt = time.time() - t0
        line = (f"epoch {epoch:3d}  loss={sum_loss/n_batches:.5f} (Y={sum_loss_y/n_batches:.5f}+"
                f"{args.chroma_weight}*UV={sum_loss_uv/n_batches:.5f})  "
                f"in={val_in/n:.3f}  enh={val_enh/n:.3f}  "
                f"ΔY={avg_gy:+.4f}  ΔU={avg_gu:+.4f}  ΔV={avg_gv:+.4f}  "
                f"lr={optim_.param_groups[0]['lr']:.1e}  ({dt:.0f}s)")
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt = f"checkpoints/{args.ckpt_prefix}_epoch_{epoch + 1}.pt"
            torch.save(model.state_dict(), ckpt)
            print(f"  saved {ckpt}", flush=True)

        # Track 'best' by the sum of per-channel gains (so chroma is rewarded)
        if avg_total_gain > best_val_gain:
            best_val_gain = avg_total_gain
            torch.save(model.state_dict(), f"checkpoints/{args.ckpt_prefix}_best.pt")


if __name__ == "__main__":
    main()
