"""
End-to-end BD-rate / BD-PSNR evaluator for VVC post-processing models.

Reads encoded bitstreams from output_eval/encoded/ and VTM-decoded YUVs +
CSV traces from output_eval/decoded/, runs a model on every (prev, curr,
next) triplet, measures Y/U/V PSNR before vs after, and computes Bjontegaard
BD-PSNR / BD-Rate over the full QP set.

Frames are processed directly from disk; we do not materialize the heavy
.pt precomputed dataset (which would cost hundreds of GB for our test set).

Usage:
    python evaluate_bd.py --model martell \\
        --checkpoint checkpoints/martell_epoch_190.pt \\
        --out bdrate_results/martell.json
"""

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from bdrate import bd_psnr, bd_rate
from features_generator.generator import FeatureMapGenerator
from features_parser.parser import VTMParser

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QPS_DEFAULT = [22, 27, 32, 37, 42]

# Same FEATURE_ORDER as scripts/precompute_features.py (martell variant, 9 ch)
FEATURE_ORDER = ["QP", "PredMode", "Depth", "Boundary",
                 "MVL0_X", "MVL0_Y", "MVL1_X", "MVL1_Y", "FrameType"]


# ---------- I/O helpers ----------

def parse_info(info_path: Path) -> dict:
    text = info_path.read_text()
    width = int(re.search(r"^Width\s*:\s+(\d+)", text, re.M).group(1))
    height = int(re.search(r"^Height\s*:\s+(\d+)", text, re.M).group(1))
    fps_match = re.search(r"^Frame rate\s+:\s+([\d.]+)", text, re.M)
    if fps_match is None:
        num = re.search(r"FrameRate_Num\s*:\s+(\d+)", text)
        den = re.search(r"FrameRate_Den\s*:\s+(\d+)", text)
        fps = float(num.group(1)) / float(den.group(1)) if num and den else 30.0
    else:
        fps = float(fps_match.group(1))
    return {"width": width, "height": height, "fps": fps}


def read_yuv_frame(f, width: int, height: int) -> np.ndarray | None:
    """Read one YUV420 frame as a (3,H,W) uint8 array with chroma upsampled."""
    y_size = width * height
    uv_size = (width // 2) * (height // 2)
    raw = f.read(y_size + 2 * uv_size)
    if len(raw) < y_size + 2 * uv_size:
        return None
    y = np.frombuffer(raw[:y_size], dtype=np.uint8).reshape(height, width)
    u = np.frombuffer(raw[y_size:y_size + uv_size], dtype=np.uint8).reshape(height // 2, width // 2)
    v = np.frombuffer(raw[y_size + uv_size:], dtype=np.uint8).reshape(height // 2, width // 2)
    return y.copy(), u.copy(), v.copy()


def read_yuv_planes(path: Path, width: int, height: int) -> tuple[np.ndarray, ...]:
    """Read all frames; returns (Y[N,H,W], U[N,H/2,W/2], V[N,H/2,W/2]) uint8."""
    Y, U, V = [], [], []
    with open(path, "rb") as f:
        while True:
            triplet = read_yuv_frame(f, width, height)
            if triplet is None:
                break
            y, u, v = triplet
            Y.append(y); U.append(u); V.append(v)
    return np.stack(Y), np.stack(U), np.stack(V)


# ---------- PSNR ----------

def psnr_uint8(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR for two uint8 arrays. Returns 100 if MSE=0."""
    a = a.astype(np.float64); b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    return 100.0 if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


# ---------- Feature generation ----------

def normalize_feature(name: str, arr: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(arr).float()
    if name == "QP":       return (t / 63.0).clamp(0, 1)
    if name == "Depth":    return (t / 7.0).clamp(0, 1)
    if name == "PredMode": return (t / 3.0).clamp(0, 1)
    if name == "Boundary": return t.clamp(0, 1)
    if name == "FrameType": return t
    if "MV" in name:       return (t / 128.0).clamp(-1, 1)
    return t


def build_feature_stack(tokens_for_poc, gen: FeatureMapGenerator) -> torch.Tensor:
    maps = gen.generate_maps_for_frame(tokens_for_poc)
    layers = []
    for name in FEATURE_ORDER:
        m = maps.get(name, np.zeros((gen.height, gen.width), dtype=np.float32))
        layers.append(normalize_feature(name, m))
    return torch.stack(layers, dim=0)  # [9, H, W]


def parse_csv_cached(csv_path: Path) -> dict:
    cache = csv_path.with_suffix(".pkl")
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    tokens = VTMParser().parse_file(str(csv_path))
    with open(cache, "wb") as f:
        pickle.dump(tokens, f)
    return tokens


# ---------- Frame conversion ----------

def to_full_res_chw(y: np.ndarray, u: np.ndarray, v: np.ndarray, height: int, width: int) -> torch.Tensor:
    """uint8 native YUV420 → float [3,H,W] in [0,1] with chroma bilinearly upsampled."""
    yt = torch.from_numpy(y).float() / 255.0
    ut = torch.from_numpy(u).float() / 255.0
    vt = torch.from_numpy(v).float() / 255.0
    u_up = F.interpolate(ut[None, None], size=(height, width), mode="bilinear", align_corners=False)[0, 0]
    v_up = F.interpolate(vt[None, None], size=(height, width), mode="bilinear", align_corners=False)[0, 0]
    return torch.stack([yt, u_up, v_up], dim=0)


def from_full_res_chw_to_planes(t: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[3,H,W] in [0,1] → (Y[H,W], U[H/2,W/2], V[H/2,W/2]) uint8 (chroma area-downsampled)."""
    y = (t[0].clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    u_full = t[1].clamp(0, 1)
    v_full = t[2].clamp(0, 1)
    h, w = y.shape
    u_half = F.avg_pool2d(u_full[None, None], 2, 2)[0, 0]
    v_half = F.avg_pool2d(v_full[None, None], 2, 2)[0, 0]
    u = (u_half * 255).round().to(torch.uint8).cpu().numpy()
    v = (v_half * 255).round().to(torch.uint8).cpu().numpy()
    return y, u, v


# ---------- Model loading ----------

def load_martell(ckpt_path: str) -> torch.nn.Module:
    from enhancer.models.snow_wide import SnowWideEnhancer
    class Cfg:
        base_channels = 64
        metadata_channels = 9
    model = SnowWideEnhancer(Cfg()).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def load_vvc_ppff(ckpt_path: str) -> torch.nn.Module:
    from enhancer.models.vvc_ppff import VVCPPFF
    model = VVCPPFF(in_channels=4, base_channels=128, num_blocks=16).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def load_snow_wide(ckpt_path: str) -> torch.nn.Module:
    from enhancer.models.snow_wide import SnowWideEnhancer
    class Cfg:
        base_channels = 64
        metadata_channels = 19
    model = SnowWideEnhancer(Cfg()).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def run_model(model: torch.nn.Module, kind: str, prev: torch.Tensor,
              curr: torch.Tensor, nxt: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
    """Returns enhanced [3,H,W] float tensor in [0,1]."""
    p = prev[None].to(DEVICE)
    c = curr[None].to(DEVICE)
    n = nxt[None].to(DEVICE)
    m = meta[None].to(DEVICE)
    with torch.no_grad():
        if kind == "vvc_ppff":
            out = model(c, m).clamp(0, 1)
        else:
            out = model(c, p, n, m).clamp(0, 1)
    return out[0]


# ---------- Eval per video ----------

def evaluate_video_qp(video: str, qp: int, args, model, kind: str) -> dict:
    width = height = None
    info = parse_info(args.orig_dir / f"{video}.y4m.info")
    width, height, fps = info["width"], info["height"], info["fps"]

    bitstream = args.bitstream_dir / f"{video}_QP{qp}.vvc"
    csv_path = args.decoded_dir / f"{video}_QP{qp}.csv"
    decoded_yuv = args.decoded_dir / f"{video}_QP{qp}_vtm_rec.yuv"
    orig_yuv = args.orig_dir / f"{video}.yuv"

    # Read YUVs (native resolution)
    Y_o, U_o, V_o = read_yuv_planes(orig_yuv, width, height)
    Y_d, U_d, V_d = read_yuv_planes(decoded_yuv, width, height)
    n_frames = min(len(Y_o), len(Y_d))
    Y_o, U_o, V_o = Y_o[:n_frames], U_o[:n_frames], V_o[:n_frames]
    Y_d, U_d, V_d = Y_d[:n_frames], U_d[:n_frames], V_d[:n_frames]

    # Anchor PSNR (native 4:2:0, averaged over frames)
    psnr_a_y = np.mean([psnr_uint8(Y_o[i], Y_d[i]) for i in range(n_frames)])
    psnr_a_u = np.mean([psnr_uint8(U_o[i], U_d[i]) for i in range(n_frames)])
    psnr_a_v = np.mean([psnr_uint8(V_o[i], V_d[i]) for i in range(n_frames)])

    # Bitrate (kbps): bytes * 8 / 1000 * fps / num_frames
    size_bytes = bitstream.stat().st_size
    bitrate_kbps = size_bytes * 8 / 1000.0 * fps / n_frames

    # Parse VTM trace once, build features per POC
    tokens_by_poc = parse_csv_cached(csv_path)
    gen = FeatureMapGenerator(width, height)

    # Pre-build float YUV tensors at full resolution (chroma upsampled) for the model
    decoded_full = [to_full_res_chw(Y_d[i], U_d[i], V_d[i], height, width) for i in range(n_frames)]

    # Run model on every (prev, curr, next) triplet
    enh_psnr_y, enh_psnr_u, enh_psnr_v = [], [], []
    for i in range(1, n_frames - 1):
        feats = build_feature_stack(tokens_by_poc.get(i, []), gen)
        enhanced = run_model(model, kind, decoded_full[i - 1], decoded_full[i], decoded_full[i + 1], feats)
        y_e, u_e, v_e = from_full_res_chw_to_planes(enhanced)
        enh_psnr_y.append(psnr_uint8(Y_o[i], y_e))
        enh_psnr_u.append(psnr_uint8(U_o[i], u_e))
        enh_psnr_v.append(psnr_uint8(V_o[i], v_e))

    # NOTE: enhanced PSNR averages over interior frames only (POCs 1..N-2),
    # while anchor PSNR averages all frames. This is consistent with how the
    # model is actually used — it can't enhance the boundary frames.
    return {
        "qp": qp,
        "n_frames": n_frames,
        "n_enhanced": len(enh_psnr_y),
        "fps": fps,
        "width": width,
        "height": height,
        "bitrate_kbps": bitrate_kbps,
        "psnr_anchor": {
            "Y": float(psnr_a_y),
            "U": float(psnr_a_u),
            "V": float(psnr_a_v),
        },
        "psnr_enhanced": {
            "Y": float(np.mean(enh_psnr_y)),
            "U": float(np.mean(enh_psnr_u)),
            "V": float(np.mean(enh_psnr_v)),
        },
        # Anchor PSNR computed on the same interior-frame slice, for an apples-to-apples
        # delta. Tiny difference vs the all-frame anchor in practice.
        "psnr_anchor_interior": {
            "Y": float(np.mean([psnr_uint8(Y_o[i], Y_d[i]) for i in range(1, n_frames - 1)])),
            "U": float(np.mean([psnr_uint8(U_o[i], U_d[i]) for i in range(1, n_frames - 1)])),
            "V": float(np.mean([psnr_uint8(V_o[i], V_d[i]) for i in range(1, n_frames - 1)])),
        },
    }


# ---------- Aggregation + BD math ----------

def aggregate_and_bd(per_video: dict, qps: list[int]) -> dict:
    """Build the model-level RD curve by averaging (rate, psnr) across videos per QP, then BD-vs-anchor."""
    agg = {qp: {"bitrate_kbps": [], "psnr_a": {"Y": [], "U": [], "V": []},
                "psnr_e": {"Y": [], "U": [], "V": []}} for qp in qps}
    for vid, by_qp in per_video.items():
        for qp in qps:
            r = by_qp.get(qp)
            if r is None:
                continue
            agg[qp]["bitrate_kbps"].append(r["bitrate_kbps"])
            for ch in ["Y", "U", "V"]:
                agg[qp]["psnr_a"][ch].append(r["psnr_anchor_interior"][ch])
                agg[qp]["psnr_e"][ch].append(r["psnr_enhanced"][ch])

    rd_mean = {qp: {} for qp in qps}
    for qp in qps:
        rd_mean[qp]["bitrate_kbps"] = float(np.mean(agg[qp]["bitrate_kbps"]))
        rd_mean[qp]["psnr_anchor"] = {ch: float(np.mean(agg[qp]["psnr_a"][ch])) for ch in "YUV"}
        rd_mean[qp]["psnr_enhanced"] = {ch: float(np.mean(agg[qp]["psnr_e"][ch])) for ch in "YUV"}

    # Build vectors for Bjontegaard
    qps_sorted = sorted(qps)
    rates = [rd_mean[qp]["bitrate_kbps"] for qp in qps_sorted]
    bd = {}
    for ch in "YUV":
        psnr_a = [rd_mean[qp]["psnr_anchor"][ch] for qp in qps_sorted]
        psnr_e = [rd_mean[qp]["psnr_enhanced"][ch] for qp in qps_sorted]
        bd[ch] = {
            "bd_psnr_db": bd_psnr(rates, psnr_a, rates, psnr_e),
            "bd_rate_pct": bd_rate(rates, psnr_a, rates, psnr_e),
        }
    return {"per_qp": rd_mean, "bd": bd}


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["martell", "snow_wide", "vvc_ppff"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--qps", default="22,27,32,37,42")
    parser.add_argument("--orig-dir", type=Path, default=Path("data_eval"))
    parser.add_argument("--bitstream-dir", type=Path, default=Path("output_eval/encoded"))
    parser.add_argument("--decoded-dir", type=Path, default=Path("output_eval/decoded"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--videos", default=None,
                        help="comma-separated video stems; defaults to all *.yuv in --orig-dir")
    args = parser.parse_args()

    qps = [int(q) for q in args.qps.split(",")]

    if args.videos:
        videos = args.videos.split(",")
    else:
        videos = sorted(p.stem for p in args.orig_dir.glob("*.yuv"))
    print(f"Videos ({len(videos)}): {videos}")
    print(f"QPs: {qps}")
    print(f"Device: {DEVICE}, Model: {args.model}, Checkpoint: {args.checkpoint}")

    loaders = {"martell": load_martell, "snow_wide": load_snow_wide, "vvc_ppff": load_vvc_ppff}
    model = loaders[args.model](args.checkpoint)

    per_video = {}
    pbar = tqdm(total=len(videos) * len(qps), desc=args.model)
    for video in videos:
        per_video[video] = {}
        for qp in qps:
            try:
                result = evaluate_video_qp(video, qp, args, model, args.model)
                per_video[video][qp] = result
                pbar.set_postfix({"v": video[:14], "qp": qp,
                                  "ΔY": f"{result['psnr_enhanced']['Y'] - result['psnr_anchor_interior']['Y']:+.3f}"})
            except Exception as e:
                print(f"\n[ERROR] {video} QP{qp}: {e}")
                per_video[video][qp] = {"error": str(e)}
            pbar.update(1)
    pbar.close()

    summary = aggregate_and_bd(per_video, qps)
    out = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "qps": qps,
        "videos": videos,
        "per_video": per_video,
        "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")
    print(json.dumps(summary["bd"], indent=2))


if __name__ == "__main__":
    main()
