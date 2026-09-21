"""Generic multi-QP retraining driver for the corrective experiment: trains ANY
of the compared architectures on VTM-encoded multi-QP data (variant "on" =
in-loop filters enabled, "off" = disabled), read on the fly by
VTMOnlineDataset from output_vtm_train/{variant}.

Usage:
    uv run python train_mqp.py --model martell_hybrid --variant off
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from gpu_lock import acquire_gpu
from vtm_online_dataset import VTMOnlineDataset
from train_martell_hybrid import y_loss  # the exact Y multi-term loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_model(name):
    if name in ("martell_hybrid",):
        from enhancer.models.snow_wide import SnowWideEnhancer

        class Cfg:
            base_channels = 64
            metadata_channels = 9
        return SnowWideEnhancer(Cfg())
    if name == "martell_hybrid_nometa":
        from enhancer.models.snow_wide_nometa import SnowWideEnhancerNoMeta

        class Cfg:
            base_channels = 64
        return SnowWideEnhancerNoMeta(Cfg())
    if name == "martell_unet":
        from enhancer.models.snow_wide_unet import SnowWideEnhancerUNet

        class Cfg:
            base_channels = 64
            unet_mid_channels = 96
            unet_bottom_channels = 128
            unet_bottleneck_blocks = 4
        return SnowWideEnhancerUNet(Cfg())
    if name == "vvc_ppff":
        from enhancer.models.vvc_ppff import VVCPPFF
        return VVCPPFF(in_channels=4, base_channels=128, num_blocks=16)
    if name == "stenet_2024":
        from enhancer.models.stenet_2024 import STENet2024

        class Cfg:
            base_channels = 64
            metadata_channels = 9
        return STENet2024(Cfg())
    if name == "bi_conv_lstm":
        from enhancer.models.bi_conv_lstm import BiConvLSTMEnhancer

        class Cfg:
            base_channels = 24
            kernel_size = 5
            cnn_layers = 5
        return BiConvLSTMEnhancer(Cfg())
    if name == "qg_conv_lstm":
        from enhancer.models.qg_conv_lstm import QGConvLSTMEnhancer

        class Cfg:
            base_channels = 64
            kernel_size = 5
            cnn_layers = 5
            metadata_channels = 9
            quality_embed = 16
        return QGConvLSTMEnhancer(Cfg())
    raise ValueError(name)


LOSS_KIND = {
    "martell_hybrid": "hybrid", "martell_hybrid_nometa": "hybrid",
    "martell_unet": "hybrid", "vvc_ppff": "mse", "stenet_2024": "stenet",
    "bi_conv_lstm": "mse", "qg_conv_lstm": "mse",
}
WEIGHT_DECAY = {"hybrid": 1e-4, "mse": 0.0, "stenet": 0.0}


def forward(name, model, prev, curr, nxt, meta):
    if name == "vvc_ppff":
        # output clamped before the loss, like in original VVC-PPFF 
        out = model(curr, meta)
        out = out[0] if isinstance(out, tuple) else out
        return out.clamp(0, 1), None
    out = model(curr, prev, nxt, meta)
    return out if isinstance(out, tuple) else (out, None)


def compute_loss(kind, enhanced, synth, original):
    if kind == "mse":
        return F.mse_loss(enhanced, original)
    if kind == "stenet":
        loss = F.mse_loss(enhanced, original)
        if synth is not None:
            loss = loss + 0.1 * F.mse_loss(synth, original)
        return loss
    ly = y_loss(enhanced[:, 0:1], original[:, 0:1])
    luv = 0.5 * (F.mse_loss(enhanced[:, 1:2], original[:, 1:2])
                 + F.mse_loss(enhanced[:, 2:3], original[:, 2:3]))
    return ly + 1.0 * luv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(LOSS_KIND))
    ap.add_argument("--variant", choices=["on", "off"], required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patch-size", type=int, default=132)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--milestones", type=str, default="13,25,38,50",
                    help="comma-separated MultiStepLR milestones")
    ap.add_argument("--tag", type=str, default="mqp",
                    help="checkpoint prefix tag: {model}_{tag}_{variant}")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore existing {prefix}_last/best checkpoints")
    ap.add_argument("--wd", type=float, default=None,
                    help="override weight decay (default: per LOSS_KIND)")
    ap.add_argument("--zero-meta", type=str, default="",
                    help="comma-separated metadata channel indices to zero "
                         "(leave-one-out ablation; order as evaluate_bd.FEATURE_ORDER)")
    args = ap.parse_args()

    prefix = f"{args.model}_{args.tag}_{args.variant}"
    zero_ch = [int(x) for x in args.zero_meta.split(",") if x.strip()]
    acquire_gpu(f"train {prefix}")
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    import random; random.seed(42); np.random.seed(42)

    man = json.load(open(Path("output_vtm_train") / args.variant
                         / "manifest_all.json"))
    train_ds = VTMOnlineDataset(man["tasks"], patch_size=args.patch_size,
                                split="train")
    val_ds = VTMOnlineDataset(man["tasks"], patch_size=args.patch_size,
                              split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False,
                            num_workers=2, pin_memory=True)

    kind = LOSS_KIND[args.model]
    model = make_model(args.model).to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters())
    wd = WEIGHT_DECAY[kind] if args.wd is None else args.wd
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=wd)
    milestones = [int(x) for x in args.milestones.split(",")]
    sched = optim.lr_scheduler.MultiStepLR(opt, milestones=milestones)
    print(f"[{prefix}] params={n_par:,} loss={kind} "
          f"wd={wd} train={len(train_ds)} val={len(val_ds)}",
          flush=True)

    os.makedirs("checkpoints", exist_ok=True)
    best = -1e9
    start_epoch = 0
    last_path = f"checkpoints/{prefix}_last.pt"
    best_path = f"checkpoints/{prefix}_best.pt"
    if not args.fresh and os.path.exists(last_path):
        st = torch.load(last_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start_epoch = st["epoch"] + 1
        best = st["best"]
        print(f"[{prefix}] resumed from {last_path} (epoch {st['epoch']}, "
              f"best {best:+.4f})", flush=True)
    elif not args.fresh and os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        print(f"[{prefix}] warm start from {best_path} (weights only)",
              flush=True)
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        tot = 0.0
        for (prev, curr, nxt), orig, meta, _ in train_loader:
            prev, curr, nxt = (x.to(DEVICE, non_blocking=True)
                               for x in (prev, curr, nxt))
            orig = orig.to(DEVICE, non_blocking=True)
            meta = meta.to(DEVICE, non_blocking=True)
            if zero_ch:
                meta[:, zero_ch] = 0
            opt.zero_grad()
            enh, synth = forward(args.model, model, prev, curr, nxt, meta)
            loss = compute_loss(kind, enh, synth, orig)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
        sched.step()
        tot /= max(1, len(train_loader))

        model.eval()
        gains = {"Y": [], "U": [], "V": []}
        with torch.no_grad():
            for (prev, curr, nxt), orig, meta, _ in val_loader:
                prev, curr, nxt = (x.to(DEVICE) for x in (prev, curr, nxt))
                orig = orig.to(DEVICE); meta = meta.to(DEVICE)
                if zero_ch:
                    meta[:, zero_ch] = 0
                enh, _s = forward(args.model, model, prev, curr, nxt, meta)
                enh = enh.clamp(0, 1)
                for i, ch in enumerate("YUV"):
                    mse_e = F.mse_loss(enh[:, i:i+1], orig[:, i:i+1])
                    mse_i = F.mse_loss(curr[:, i:i+1], orig[:, i:i+1])
                    g = (10 * torch.log10(1.0 / mse_e)
                         - 10 * torch.log10(1.0 / mse_i))
                    gains[ch].append(float(g))
        gy, gu, gv = (float(np.mean(gains[c])) for c in "YUV")
        total_gain = gy + gu + gv
        line = (f"epoch {epoch:3d}  loss={tot:.5f}  "
                f"dY={gy:+.4f} dU={gu:+.4f} dV={gv:+.4f}  "
                f"lr={sched.get_last_lr()[0]:.1e}  ({time.time()-t0:.0f}s)")
        print(line, flush=True)
        if total_gain > best:
            best = total_gain
            torch.save(model.state_dict(), f"checkpoints/{prefix}_best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch,
                    "best": best}, last_path)
    print(f"[{prefix}] done, best total gain {best:+.4f} dB", flush=True)


if __name__ == "__main__":
    main()
