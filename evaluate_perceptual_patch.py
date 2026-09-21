"""Patch-based perceptual eval: same 132x132 grid as evaluate_patch.py,
same metrics as evaluate_perceptual.py.
"""

import argparse
import json
import warnings
from pathlib import Path

import lpips
import numpy as np
import torch
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

PATCH_DEFAULT = 132


def patch_grid(h, w, p):
    rows = list(range(0, h - p + 1, p))
    cols = list(range(0, w - p + 1, p))
    return [(r, c) for r in rows for c in cols]


def y_uint8_to_3ch_01(y):
    """uint8 [H,W] -> float [1,3,H,W] in [0,1]."""
    t = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0) / 255.0
    return t.expand(-1, 3, -1, -1)


def y_uint8_to_3ch_m11(y):
    """uint8 [H,W] -> float [1,3,H,W] in [-1,1] for LPIPS."""
    t = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0) / 127.5 - 1.0
    return t.expand(-1, 3, -1, -1)


def evaluate_video_qp(video, qp, args, model, kind, lpips_alex, lpips_vgg):
    PATCH = args.patch_size
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
    original_full = [to_full_res_chw(Y_o[i], U_o[i], V_o[i], height, width)
                     for i in range(n_frames)]

    coords = patch_grid(height, width, PATCH)

    metrics = {"a_ssim": [], "e_ssim": [],
               "a_msssim": [], "e_msssim": [],
               "a_lpips_a": [], "e_lpips_a": [],
               "a_lpips_v": [], "e_lpips_v": []}

    for i in range(1, n_frames - 1):
        feats_full = build_feature_stack(tokens_by_poc.get(i, []), gen)
        for (r, c) in coords:
            sl = (slice(None), slice(r, r + PATCH), slice(c, c + PATCH))
            prev_p = decoded_full[i - 1][sl]
            curr_p = decoded_full[i][sl]
            next_p = decoded_full[i + 1][sl]
            orig_p = original_full[i][sl]
            feats_p = feats_full[sl]

            enhanced_p = run_model(model, kind, prev_p, curr_p, next_p, feats_p)

            ya, _, _ = from_full_res_chw_to_planes(curr_p)
            ye, _, _ = from_full_res_chw_to_planes(enhanced_p)
            yo, _, _ = from_full_res_chw_to_planes(orig_p)

            to_o = y_uint8_to_3ch_01(yo).to(DEVICE)
            to_a = y_uint8_to_3ch_01(ya).to(DEVICE)
            to_e = y_uint8_to_3ch_01(ye).to(DEVICE)

            with torch.no_grad():
                metrics["a_ssim"].append(float(ssim(to_a, to_o, data_range=1.0, size_average=True)))
                metrics["e_ssim"].append(float(ssim(to_e, to_o, data_range=1.0, size_average=True)))
                metrics["a_msssim"].append(float(ms_ssim(to_a, to_o, data_range=1.0, size_average=True, win_size=7)))
                metrics["e_msssim"].append(float(ms_ssim(to_e, to_o, data_range=1.0, size_average=True, win_size=7)))

                lp_o = y_uint8_to_3ch_m11(yo).to(DEVICE)
                lp_a = y_uint8_to_3ch_m11(ya).to(DEVICE)
                lp_e = y_uint8_to_3ch_m11(ye).to(DEVICE)
                metrics["a_lpips_a"].append(float(lpips_alex(lp_a, lp_o).squeeze()))
                metrics["e_lpips_a"].append(float(lpips_alex(lp_e, lp_o).squeeze()))
                metrics["a_lpips_v"].append(float(lpips_vgg(lp_a, lp_o).squeeze()))
                metrics["e_lpips_v"].append(float(lpips_vgg(lp_e, lp_o).squeeze()))

    n_patches = len(metrics["a_ssim"])
    return {
        "qp": qp, "n_frames": n_frames, "n_patches": n_patches,
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
    agg = {"anchor":   {k: [] for k in ["ssim", "msssim", "lpips_alex", "lpips_vgg"]},
           "enhanced": {k: [] for k in ["ssim", "msssim", "lpips_alex", "lpips_vgg"]}}
    per_qp = {qp: {"anchor":   {k: [] for k in agg["anchor"]},
                   "enhanced": {k: [] for k in agg["anchor"]}}
              for qp in qps}
    for _vid, by_qp in per_video.items():
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
    summary["overall"]["delta"] = {k: summary["overall"]["enhanced"][k] - summary["overall"]["anchor"][k]
                                   for k in agg["anchor"]}
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["martell", "martell_nometa", "martell_unet",
                                            "snow_wide",
                                            "vvc_ppff", "stenet", "bi_conv_lstm",
                                            "qg_conv_lstm"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--qps", default="22,27,32,37,42")
    parser.add_argument("--orig-dir", type=Path, default=Path("data_eval"))
    parser.add_argument("--decoded-dir", type=Path, default=Path("output_eval/decoded"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--videos", default=None,
                        help="comma-separated; defaults to all *.yuv in --orig-dir")
    parser.add_argument("--patch-size", type=int, default=PATCH_DEFAULT,
                        help="non-overlapping patch size (px); default 132 to match training")
    args = parser.parse_args()

    qps = [int(q) for q in args.qps.split(",")]
    if args.videos:
        videos = args.videos.split(",")
    else:
        videos = sorted(p.stem for p in args.orig_dir.glob("*.yuv"))
    print(f"Patch perceptual eval ({args.patch_size}x{args.patch_size}). Model: {args.model}  Videos: {len(videos)}  QPs: {qps}")

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
                r = evaluate_video_qp(video, qp, args, model, args.model, lpips_alex, lpips_vgg)
                per_video[video][qp] = r
                d_ssim = r["enhanced"]["ssim"] - r["anchor"]["ssim"]
                d_lpa  = r["enhanced"]["lpips_alex"] - r["anchor"]["lpips_alex"]
                pbar.set_postfix({"v": video[:14], "qp": qp,
                                  "Δssim": f"{d_ssim:+.4f}",
                                  "Δlpips": f"{d_lpa:+.4f}"})
            except Exception as e:
                print(f"\n[ERROR] {video} QP{qp}: {e}")
                per_video[video][qp] = {"error": str(e)}
            pbar.update(1)
    pbar.close()

    summary = aggregate(per_video, qps)
    out = {"model": args.model, "checkpoint": args.checkpoint, "qps": qps,
           "patch_size": args.patch_size, "videos": videos,
           "per_video": per_video, "summary": summary}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()
