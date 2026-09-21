"""Train STENet (paper architecture, see enhancer/models/stenet_2024.py)
on the same 9-ch precomputed_martell dataset, using the same
optimiser/schedule/loss as VVC-PPFF and Martell-MSE so the comparison
on the unbiased test set is apples-to-apples.
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio

from enhancer.dataset_blackfyre import BlackfyreDataset
from enhancer.models.stenet_2024 import STENet2024

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patch-size", type=int, default=132)
    p.add_argument("--data-dir", default="data/precomputed_martell")
    p.add_argument("--ckpt-prefix", default="stenet_2024")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--synth-weight", type=float, default=0.1)
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

    model = STENet2024(Cfg()).to(DEVICE)
    print(f"Params: {sum(x.numel() for x in model.parameters()):,}")

    # no weight decay: with a residual MSE loss it drives the conv weights to zero
    optim_ = optim.Adam(model.parameters(), lr=args.lr)
    sched = optim.lr_scheduler.MultiStepLR(optim_, milestones=[50, 100, 150, 200, 300])
    psnr_e = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    psnr_i = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)

    os.makedirs("checkpoints", exist_ok=True)
    log_path = f"{args.ckpt_prefix}_train.log"
    print(f"Logging to {log_path}")

    best_val_gain = -1e9
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_loss_main = 0.0
        train_loss_synth = 0.0

        for decoded, original, metadata, _info in train_loader:
            prev_frame, curr_frame, next_frame = decoded
            prev_frame = prev_frame.to(DEVICE, non_blocking=True)
            curr_frame = curr_frame.to(DEVICE, non_blocking=True)
            next_frame = next_frame.to(DEVICE, non_blocking=True)
            original = original.to(DEVICE, non_blocking=True)
            metadata = metadata.to(DEVICE, non_blocking=True)

            optim_.zero_grad()
            enhanced, synth = model(curr_frame, prev_frame, next_frame, metadata)
            loss_main = F.mse_loss(enhanced, original)
            loss_synth = F.mse_loss(synth, original)
            loss = loss_main + args.synth_weight * loss_synth
            loss.backward()
            optim_.step()
            train_loss += float(loss.detach())
            train_loss_main += float(loss_main.detach())
            train_loss_synth += float(loss_synth.detach())

        sched.step()
        train_loss /= len(train_loader)
        train_loss_main /= len(train_loader)
        train_loss_synth /= len(train_loader)

        # Validation
        model.eval()
        val_gain = val_in = val_enh = 0.0; n = 0
        val_gain_y = val_gain_u = val_gain_v = 0.0
        with torch.no_grad():
            for decoded, original, metadata, _info in val_loader:
                prev_frame, curr_frame, next_frame = decoded
                prev_frame = prev_frame.to(DEVICE, non_blocking=True)
                curr_frame = curr_frame.to(DEVICE, non_blocking=True)
                next_frame = next_frame.to(DEVICE, non_blocking=True)
                original = original.to(DEVICE, non_blocking=True)
                metadata = metadata.to(DEVICE, non_blocking=True)
                enhanced, _ = model(curr_frame, prev_frame, next_frame, metadata)
                for i in range(enhanced.shape[0]):
                    pe = psnr_e(enhanced[i:i+1], original[i:i+1]).item()
                    pi = psnr_i(curr_frame[i:i+1], original[i:i+1]).item()
                    val_enh += pe; val_in += pi; val_gain += (pe - pi); n += 1
                    for ch_idx in range(3):
                        mse_e = F.mse_loss(enhanced[i:i+1, ch_idx:ch_idx+1],
                                           original[i:i+1, ch_idx:ch_idx+1]).item()
                        mse_in = F.mse_loss(curr_frame[i:i+1, ch_idx:ch_idx+1],
                                            original[i:i+1, ch_idx:ch_idx+1]).item()
                        d = (10 * np.log10(1.0 / max(mse_e, 1e-12))
                             - 10 * np.log10(1.0 / max(mse_in, 1e-12)))
                        if ch_idx == 0: val_gain_y += d
                        elif ch_idx == 1: val_gain_u += d
                        else: val_gain_v += d
                    psnr_e.reset(); psnr_i.reset()

        avg_gain = val_gain / n
        avg_gy = val_gain_y / n; avg_gu = val_gain_u / n; avg_gv = val_gain_v / n
        dt = time.time() - t0
        line = (f"epoch {epoch:3d}  loss={train_loss:.5f} (main={train_loss_main:.5f}+synth={train_loss_synth:.5f})  "
                f"in={val_in/n:.3f}  enh={val_enh/n:.3f}  Δ={avg_gain:+.4f}  "
                f"ΔY={avg_gy:+.4f}  ΔU={avg_gu:+.4f}  ΔV={avg_gv:+.4f}  "
                f"lr={optim_.param_groups[0]['lr']:.1e}  ({dt:.0f}s)")
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt = f"checkpoints/{args.ckpt_prefix}_epoch_{epoch + 1}.pt"
            torch.save(model.state_dict(), ckpt)
            print(f"  saved {ckpt}", flush=True)
        if avg_gain > best_val_gain:
            best_val_gain = avg_gain
            torch.save(model.state_dict(), f"checkpoints/{args.ckpt_prefix}_best.pt")


if __name__ == "__main__":
    main()
