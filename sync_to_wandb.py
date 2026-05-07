"""
Post-hoc sync of training runs to W&B by parsing the text logs my
training drivers produced (martell_mse_train.log, martell_hybrid_train.log,
etc.). I didn't bake wandb into those drivers, so this is the only
way to reconstruct runs without re-training.

Each log line looks like:
  epoch  50  loss=0.04226 (Y=...)  in=37.080  enh=37.547  ΔY=+0.6896  ΔU=+0.0100  ΔV=+0.0232  lr=1.0e-05  (62s)

Usage:
    python sync_to_wandb.py --log martell_hybrid_train.log --name martell-hybrid --project vvc-cnn-inter
    python sync_to_wandb.py --all
"""

import argparse
import re
from pathlib import Path

import wandb


# Tolerant regex — matches both train_martell_mse / train_snow_wide_9ch /
# train_stenet (which has a different loss-decomposition format) and
# train_martell_hybrid logs.
LINE_RE = re.compile(
    r"epoch\s+(?P<epoch>\d+)\s+"
    r"loss=(?P<loss>[\d.e+-]+)"
    r"(?:\s*\([^)]*\))?\s+"
    r"in=(?P<vin>[\d.]+)\s+"
    r"enh=(?P<venh>[\d.]+)\s+"
    r"(?:Δ=(?P<dtotal>[+-]?[\d.]+)\s+)?"
    r"ΔY=(?P<dy>[+-]?[\d.]+)\s+"
    r"ΔU=(?P<du>[+-]?[\d.]+)\s+"
    r"ΔV=(?P<dv>[+-]?[\d.]+)\s+"
    r"lr=(?P<lr>[\d.e+-]+)"
)


def parse_log(path: Path):
    out = []
    for line in path.read_text().splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        d = m.groupdict()
        out.append({
            "epoch": int(d["epoch"]),
            "train_loss": float(d["loss"]),
            "val_psnr_input": float(d["vin"]),
            "val_psnr_enhanced": float(d["venh"]),
            "val_gain_total": (float(d["dtotal"]) if d["dtotal"] is not None
                               else float(d["dy"]) + float(d["du"]) + float(d["dv"])),
            "val_gain_y": float(d["dy"]),
            "val_gain_u": float(d["du"]),
            "val_gain_v": float(d["dv"]),
            "lr": float(d["lr"]),
        })
    return out


# Mapping of log files to wandb run names + descriptive notes.
RUNS = [
    ("martell_mse_train.log", "martell-mse",
     "Pure MSE retraining of Martell architecture. "
     "Same data + optimiser + schedule as original Martell training, "
     "loss replaced with F.mse_loss(enhanced, original)."),
    ("martell_hybrid_train.log", "martell-hybrid",
     "Hybrid loss: original Martell multi-term loss restricted to the Y "
     "plane only, plus pure MSE on chroma (chroma_weight=1.0)."),
    ("martell_hybrid_ft_train.log", "martell-hybrid-ft",
     "Fine-tune of martell_hybrid_best.pt with chroma_weight=50, lr=1e-5, "
     "60 epochs — try to push chroma improvement without losing Y."),
    ("snow_wide_9ch_train.log", "snow-wide-9ch",
     "Snow-Wide retrained with metadata_channels=9 (matching Martell). "
     "Original 19-channel feature recipe was unrecoverable."),
    ("stenet_2024_train.log", "stenet-2024",
     "STENet (2024 paper architecture: RFS + PFE). MSE + 0.1*MSE on synth."),
]


def sync_run(log_path: Path, name: str, notes: str, project: str, mode: str):
    rows = parse_log(log_path)
    if not rows:
        print(f"  no parsable epoch lines in {log_path} — skipping")
        return
    run = wandb.init(project=project, name=name, notes=notes, mode=mode, reinit=True)
    for row in rows:
        wandb.log(row, step=row["epoch"])
    run.finish()
    print(f"  synced {len(rows)} epochs as run '{name}'")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, default=None,
                   help="single log file to sync; default = sync all known runs")
    p.add_argument("--name", default=None)
    p.add_argument("--notes", default="")
    p.add_argument("--project", default="vvc-cnn-inter")
    p.add_argument("--mode", default="online", choices=["online", "offline"])
    args = p.parse_args()

    if args.log is not None:
        if not args.log.exists():
            print(f"log not found: {args.log}")
            return
        sync_run(args.log, args.name or args.log.stem, args.notes, args.project, args.mode)
        return

    for fname, name, notes in RUNS:
        path = Path(fname)
        if not path.exists():
            print(f"skip {fname} (not present)")
            continue
        print(f"syncing {fname} -> {name}")
        sync_run(path, name, notes, args.project, args.mode)


if __name__ == "__main__":
    main()
