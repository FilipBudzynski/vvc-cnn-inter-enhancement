"""
Perceptual quality evaluation: SSIM, MS-SSIM, LPIPS (Alex + VGG).

Runs each enhancement model on the same (video, QP) triplets used by
evaluate_bd.py and computes perceptual metrics on the Y channel
(replicated to 3 channels for LPIPS/MS-SSIM which expect RGB).

Anchor metrics are computed against the unfiltered VTM reconstruction;
enhanced metrics against the model output. We report mean over all
(frame, QP, video) combinations, and the per-QP averages.

For SSIM/MS-SSIM higher is better (range ~[0,1]).
For LPIPS lower is better: it's a perceptual distance, not similarity.

Output: bdrate_results/<model>_perceptual.json

Usage:
    uv run python evaluate_perceptual.py --model martell \\
        --checkpoint checkpoints/martell_hybrid_best.pt \\
        --out bdrate_results/martell_hybrid_perceptual.json \\
        --videos Johnny_1280x720_60,vidyo1_720p_60fps,vidyo3_720p_60fps
"""

import argparse
import json
import warnings
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim
from tqdm import tqdm

from evaluate_bd import (
    DEVICE, build_feature_stack, from_full_res_chw_to_planes,
    load_bi_conv_lstm, load_martell, load_martell_nometa, load_martell_unet, load_qg_conv_lstm,
    load_snow_wide, load_stenet, load_vvc_ppff, parse_csv_cached, parse_info,
    read_yuv_planes, run_model, to_full_res_chw,
)
from features_generator.generator import FeatureMapGenerator

warnings.filterwarnings("ignore", category=UserWarning)


def y_to_3ch(y_uint8: np.ndarray) -> torch.Tensor:
    """uint8 Y plane [H,W] -> float [1,3,H,W] in [0,1]."""
    t = torch.from_numpy(y_uint8).float().unsqueeze(0).unsqueeze(0) / 255.0
    return t.expand(-1, 3, -1, -1)


def y_to_lpips(y_uint8: np.ndarray) -> torch.Tensor:
    """uint8 Y plane [H,W] -> float [1,3,H,W] in [-1,1] (LPIPS expects [-1,1])."""
    t = torch.from_numpy(y_uint8).float().unsqueeze(0).unsqueeze(0) / 127.5 - 1.0
    return t.expand(-1, 3, -1, -1)


def evaluate_video_qp_perceptual(video, qp, args, model, kind,
                                 lpips_alex, lpips_vgg):
    info = parse_info(args.orig_dir / f"{video}.y4m.info")
    width, height = info["width"], info["height"]

    csv_path = args.decoded_dir / f"{video}_QP{qp}.csv"
    decoded_yuv = args.decoded_dir / f"{video}_QP{qp}_vtm_rec.yuv"
    orig_yuv = args.orig_dir / f"{video}.yuv"

    Y_o, U_o, V_o = read_yuv_planes(orig_yuv, width, height)
    Y_d, U_d, V_d = read_yuv_planes(decoded_yuv, width, height)
    n_frames = min(len(Y_o), len(Y_d))

    tokens_by_poc = parse_csv_cached(csv_path)
    gen = FeatureMapGenerator(width, height)
    decoded_full = [to_full_res_chw(Y_d[i], U_d[i], V_d[i], height, width)
                    for i in range(n_frames)]

    metrics = {"a_ssim": [], "e_ssim": [],
               "a_msssim": [], "e_msssim": [],
               "a_lpips_a": [], "e_lpips_a": [],
               "a_lpips_v": [], "e_lpips_v": []}

    for i in range(1, n_frames - 1):
        feats = build_feature_stack(tokens_by_poc.get(i, []), gen)
        enhanced_chw = run_model(model, kind, decoded_full[i - 1],
                                 decoded_full[i], decoded_full[i + 1], feats)
        y_e, _, _ = from_full_res_chw_to_planes(enhanced_chw)

        y_o = Y_o[i]  # original Y, uint8
        y_a = Y_d[i]  # anchor Y, uint8 (VTM reconstruction)

        # SSIM / MS-SSIM use [0,1] 3ch tensors (replicate Y)
        to_o = y_to_3ch(y_o).to(DEVICE)
        to_a = y_to_3ch(y_a).to(DEVICE)
        to_e = y_to_3ch(y_e).to(DEVICE)

        with torch.no_grad():
            metrics["a_ssim"].append(float(ssim(to_a, to_o, data_range=1.0, size_average=True)))
            metrics["e_ssim"].append(float(ssim(to_e, to_o, data_range=1.0, size_average=True)))
            metrics["a_msssim"].append(float(ms_ssim(to_a, to_o, data_range=1.0, size_average=True, win_size=7)))
            metrics["e_msssim"].append(float(ms_ssim(to_e, to_o, data_range=1.0, size_average=True, win_size=7)))

            # LPIPS uses [-1,1] 3ch tensors
            lp_o = y_to_lpips(y_o).to(DEVICE)
            lp_a = y_to_lpips(y_a).to(DEVICE)
            lp_e = y_to_lpips(y_e).to(DEVICE)
            metrics["a_lpips_a"].append(float(lpips_alex(lp_a, lp_o).squeeze()))
            metrics["e_lpips_a"].append(float(lpips_alex(lp_e, lp_o).squeeze()))
            metrics["a_lpips_v"].append(float(lpips_vgg(lp_a, lp_o).squeeze()))
            metrics["e_lpips_v"].append(float(lpips_vgg(lp_e, lp_o).squeeze()))

    return {
        "qp": qp,
        "n_frames": n_frames,
        "n_enhanced": len(metrics["a_ssim"]),
        "anchor":   {"ssim":  float(np.mean(metrics["a_ssim"])),
                     "msssim": float(np.mean(metrics["a_msssim"])),
                     "lpips_alex": float(np.mean(metrics["a_lpips_a"])),
                     "lpips_vgg":  float(np.mean(metrics["a_lpips_v"]))},
        "enhanced": {"ssim":  float(np.mean(metrics["e_ssim"])),
                     "msssim": float(np.mean(metrics["e_msssim"])),
                     "lpips_alex": float(np.mean(metrics["e_lpips_a"])),
                     "lpips_vgg":  float(np.mean(metrics["e_lpips_v"]))},
    }


def aggregate(per_video, qps):
    """Mean over (video, QP). Anchor and enhanced separately."""
    agg = {"anchor":   {k: [] for k in ["ssim", "msssim", "lpips_alex", "lpips_vgg"]},
           "enhanced": {k: [] for k in ["ssim", "msssim", "lpips_alex", "lpips_vgg"]}}
    per_qp = {qp: {"anchor": {k: [] for k in ["ssim", "msssim", "lpips_alex", "lpips_vgg"]},
                   "enhanced": {k: [] for k in ["ssim", "msssim", "lpips_alex", "lpips_vgg"]}}
              for qp in qps}
    for vid, by_qp in per_video.items():
        for qp in qps:
            r = by_qp.get(qp)
            if r is None or "error" in r:
                continue
            for k in agg["anchor"]:
                agg["anchor"][k].append(r["anchor"][k])
                agg["enhanced"][k].append(r["enhanced"][k])
                per_qp[qp]["anchor"][k].append(r["anchor"][k])
                per_qp[qp]["enhanced"][k].append(r["enhanced"][k])

    summary = {"overall": {"anchor":   {k: float(np.mean(v)) for k, v in agg["anchor"].items()},
                           "enhanced": {k: float(np.mean(v)) for k, v in agg["enhanced"].items()}},
               "per_qp": {qp: {"anchor":   {k: float(np.mean(v)) for k, v in per_qp[qp]["anchor"].items()},
                               "enhanced": {k: float(np.mean(v)) for k, v in per_qp[qp]["enhanced"].items()}}
                          for qp in qps}}
    # Delta = enhanced - anchor (positive = improvement for SSIM/MS-SSIM, negative = improvement for LPIPS)
    summary["overall"]["delta"] = {k: summary["overall"]["enhanced"][k] - summary["overall"]["anchor"][k]
                                   for k in agg["anchor"]}
    return summary


def main():
    from gpu_lock import acquire_gpu
    acquire_gpu("evaluate_perceptual")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["martell", "martell_nometa", "martell_unet",
                                            "snow_wide",
                                            "vvc_ppff", "stenet", "bi_conv_lstm",
                                            "qg_conv_lstm"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--qps", default="22,27,32,37,42")
    parser.add_argument("--orig-dir", type=Path, default=Path("data_eval"))
    parser.add_argument("--bitstream-dir", type=Path, default=Path("output_eval/encoded"))
    parser.add_argument("--decoded-dir", type=Path, default=Path("output_eval/decoded"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--videos", default="Johnny_1280x720_60,vidyo1_720p_60fps,vidyo3_720p_60fps",
                        help="comma-separated video stems")
    args = parser.parse_args()

    qps = [int(q) for q in args.qps.split(",")]
    videos = args.videos.split(",")
    print(f"Perceptual eval: {args.model} on {videos}  QPs: {qps}")

    loaders = {"martell": load_martell, "martell_nometa": load_martell_nometa,
               "martell_unet": load_martell_unet,
               "snow_wide": load_snow_wide,
               "vvc_ppff": load_vvc_ppff, "stenet": load_stenet,
               "bi_conv_lstm": load_bi_conv_lstm, "qg_conv_lstm": load_qg_conv_lstm}
    model = loaders[args.model](args.checkpoint)

    lpips_alex = lpips.LPIPS(net="alex", verbose=False).to(DEVICE).eval()
    lpips_vgg = lpips.LPIPS(net="vgg", verbose=False).to(DEVICE).eval()

    per_video = {}
    pbar = tqdm(total=len(videos) * len(qps), desc=args.model)
    for video in videos:
        per_video[video] = {}
        for qp in qps:
            try:
                r = evaluate_video_qp_perceptual(video, qp, args, model, args.model,
                                                 lpips_alex, lpips_vgg)
                per_video[video][qp] = r
                pbar.set_postfix({"v": video[:14], "qp": qp,
                                  "Δssim": f"{r['enhanced']['ssim']-r['anchor']['ssim']:+.4f}",
                                  "Δlpips": f"{r['enhanced']['lpips_alex']-r['anchor']['lpips_alex']:+.4f}"})
            except Exception as e:
                print(f"\n[ERROR] {video} QP{qp}: {e}")
                per_video[video][qp] = {"error": str(e)}
            pbar.update(1)
    pbar.close()

    summary = aggregate(per_video, qps)
    out = {"model": args.model, "checkpoint": args.checkpoint, "qps": qps,
           "videos": videos, "per_video": per_video, "summary": summary}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
