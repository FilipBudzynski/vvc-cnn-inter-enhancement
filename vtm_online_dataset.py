"""On-the-fly multi-QP training dataset for the corrective experiment."""

import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch

import evaluate_bd as E


def _poc_cache_path(csv_path: Path, poc: int) -> Path:
    return csv_path.with_suffix(f".poc{poc:03d}.pkl")


def _tokens_for_poc(csv_path: Path, poc: int):
    """Small per-POC token pickle; built once from the full parse, atomically."""
    pc = _poc_cache_path(csv_path, poc)
    if pc.exists():
        with open(pc, "rb") as f:
            return pickle.load(f)
    tokens = E.parse_csv_cached(csv_path)  # full parse (disk-cached .pkl)
    for p, toks in tokens.items():
        out = _poc_cache_path(csv_path, p)
        if out.exists():
            continue
        tmp = out.with_suffix(out.suffix + f".tmp{os.getpid()}")
        with open(tmp, "wb") as f:
            pickle.dump(toks, f)
        os.replace(tmp, out)
    if not pc.exists():  # POC absent from the trace: cache the empty list too
        tmp = pc.with_suffix(pc.suffix + f".tmp{os.getpid()}")
        with open(tmp, "wb") as f:
            pickle.dump([], f)
        os.replace(tmp, pc)
    return tokens.get(poc, [])


def _paint_crop(tokens, top, left, th, tw):
    """9-channel feature stack for the crop window only."""
    maps = {}

    def canvas(name):
        if name not in maps:
            maps[name] = np.zeros((th, tw), dtype=np.float32)
        return maps[name]

    n_intra = n_inter = n_pred = 0
    for t in tokens:
        if t.param == "PredMode":
            n_pred += 1
            if t.value == 1.0:
                n_intra += 1
            elif t.value in (2.0, 3.0):
                n_inter += 1
        x0 = max(t.x - left, 0); y0 = max(t.y - top, 0)
        x1 = min(t.x + t.w - left, tw); y1 = min(t.y + t.h - top, th)
        if x0 >= x1 or y0 >= y1:
            continue
        if hasattr(t.value, "x"):  # VectorToken
            canvas(f"{t.param}_X")[y0:y1, x0:x1] = t.value.x
            canvas(f"{t.param}_Y")[y0:y1, x0:x1] = t.value.y
        else:
            canvas(t.param)[y0:y1, x0:x1] = t.value
        if "Depth" in t.param:
            b = canvas("Boundary")
            if 0 <= t.y - top < th:            # real top edge inside window
                b[t.y - top, x0:x1] = 1.0
            if 0 <= t.x - left < tw:           # real left edge inside window
                b[y0:y1, t.x - left] = 1.0
    if n_pred:
        intra_ratio = n_intra / n_pred
        if intra_ratio > 0.9:
            ft = 0.0
        elif n_inter > n_intra:
            ft = 1.0
        else:
            ft = 0.5
        maps["FrameType"] = np.full((th, tw), ft, dtype=np.float32)

    layers = []
    for name in E.FEATURE_ORDER:
        m = maps.get(name, np.zeros((th, tw), dtype=np.float32))
        layers.append(E.normalize_feature(name, m))
    return torch.stack(layers, dim=0)  # [9, th, tw]


class VTMOnlineDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_tasks, patch_size=132, split="train",
                 train_ratio=0.9, seed=42, frames=64):
        self.patch = patch_size
        self.frames = frames
        self.tasks = [t for t in manifest_tasks
                      if t["height"] >= patch_size and t["width"] >= patch_size]
        self.samples = []
        for ti, t in enumerate(self.tasks):
            for poc in range(1, frames - 1):
                self.samples.append((ti, poc))
        rng = random.Random(seed)
        rng.shuffle(self.samples)
        n = len(self.samples)
        cut = int(n * train_ratio)
        self.samples = self.samples[:cut] if split == "train" else self.samples[cut:]
        print(f"[VTMOnlineDataset] {split}: {len(self.samples)} triplets "
              f"from {len(self.tasks)} (video,QP) pairs")

    def __len__(self):
        return len(self.samples)

    def _read_crop(self, path, poc, w, h, top, left, th, tw):
        """Crop-only YUV420 read + local bilinear chroma upsample (8-px
        aligned crop keeps the interpolation grid phase identical)."""
        fb = w * h * 3 // 2
        mm = np.memmap(path, dtype=np.uint8, mode="r",
                       offset=poc * fb, shape=(fb,))
        y = np.array(mm[:w * h].reshape(h, w)[top:top + th, left:left + tw])
        uo = w * h
        u = np.array(mm[uo:uo + w * h // 4].reshape(h // 2, w // 2)
                     [top // 2:(top + th) // 2, left // 2:(left + tw) // 2])
        vo = uo + w * h // 4
        v = np.array(mm[vo:vo + w * h // 4].reshape(h // 2, w // 2)
                     [top // 2:(top + th) // 2, left // 2:(left + tw) // 2])
        return E.to_full_res_chw(y, u, v, th, tw)

    def __getitem__(self, idx):
        ti, poc = self.samples[idx]
        t = self.tasks[ti]
        w, h = t["width"], t["height"]
        th = tw = self.patch
        top = random.choice(range(0, h - th + 1, 8)) if h > th else 0
        left = random.choice(range(0, w - tw + 1, 8)) if w > tw else 0

        dec = t["vtm_rec"]
        prev = self._read_crop(dec, poc - 1, w, h, top, left, th, tw)
        curr = self._read_crop(dec, poc, w, h, top, left, th, tw)
        nxt = self._read_crop(dec, poc + 1, w, h, top, left, th, tw)
        orig = self._read_crop(t["input"], poc, w, h, top, left, th, tw)

        toks = _tokens_for_poc(Path(t["csv"]), poc)
        feats = _paint_crop(toks, top, left, th, tw)

        return ((prev, curr, nxt), orig, feats,
                {"video": t["video"], "poc": poc, "qp": t["qp"]})
