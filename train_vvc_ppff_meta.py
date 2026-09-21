"""Train VVC-PPFF-Meta: VVC-PPFF backbone with Martell-style 9-ch decoder
metadata (input concat + late metadata gate).
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from enhancer.dataset_blackfyre import BlackfyreDataset
from enhancer.models.vvc_ppff_meta import VVCPPFFMeta

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patch-size", type=int, default=132)
    p.add_argument("--data-dir", default="data/precomputed_martell")
    p.add_argument("--ckpt-prefix", default="vvc_ppff_meta")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-every", type=int, default=20)
    p.add_argument("--base-channels", type=int, default=128)
    p.add_argument("--num-blocks", type=int, default=16)
    args = p.parse_args()

    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    import random; random.seed(42); np.random.seed(42)

    train_ds = BlackfyreDataset(args.data_dir, patch_size=args.patch_size, split="train")
    val_ds = BlackfyreDataset(args.data_dir, patch_size=args.patch_size, split="val")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = VVCPPFFMeta(in_channels=3, metadata_channels=9,
                        base_channels=args.base_channels, num_blocks=args.num_blocks).to(DEVICE)
    print(f"Params: {sum(x.numel() for x in model.parameters()):,}")

    optim_ = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.MultiStepLR(optim_, milestones=[50, 100, 150, 200, 300])

    os.makedirs("checkpoints", exist_ok=True)
    log_path = f"{args.ckpt_prefix}_train.log"
    print(f"Logging to {log_path}")

    best_val_gain = -1e9
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for decoded, original, metadata, _info in train_loader:
            _prev, curr_frame, _next = decoded  # VVC-PPFF is single-frame
            curr_frame = curr_frame.to(DEVICE, non_blocking=True)
            original = original.to(DEVICE, non_blocking=True)
            metadata = metadata.to(DEVICE, non_blocking=True)

            optim_.zero_grad()
            enhanced = model(curr_frame, metadata)
            loss = F.mse_loss(enhanced, original)
            loss.backward()
            optim_.step()
            train_loss += float(loss.detach())

        sched.step()
        train_loss /= len(train_loader)

        model.eval()
        val_in = val_enh = 0.0; val_gy = val_gu = val_gv = 0.0; n = 0
        with torch.no_grad():
            for decoded, original, metadata, _info in val_loader:
                _prev, curr_frame, _next = decoded
                curr_frame = curr_frame.to(DEVICE, non_blocking=True)
                original = original.to(DEVICE, non_blocking=True)
                metadata = metadata.to(DEVICE, non_blocking=True)
                enhanced = model(curr_frame, metadata)
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
        avg_total = avg_gy + avg_gu + avg_gv
        dt = time.time() - t0
        line = (f"epoch {epoch:3d}  loss={train_loss:.5f}  "
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
        if avg_total > best_val_gain:
            best_val_gain = avg_total
            torch.save(model.state_dict(), f"checkpoints/{args.ckpt_prefix}_best.pt")


if __name__ == "__main__":
    main()
