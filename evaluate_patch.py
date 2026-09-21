"""Patch-level evaluation for all enhancement models."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from bdrate import bd_psnr, bd_rate
from evaluate_bd import (
    DEVICE, FEATURE_ORDER, build_feature_stack, from_full_res_chw_to_planes,
    load_bi_conv_lstm, load_martell, load_martell_nometa, load_martell_unet, load_qg_conv_lstm,
    load_snow_wide, load_stenet, load_vvc_ppff, parse_csv_cached, parse_info,
    psnr_uint8, read_yuv_planes, run_model, to_full_res_chw,
)
from features_generator.generator import FeatureMapGenerator

PATCH_DEFAULT = 132
QPS_DEFAULT = [22, 27, 32, 37, 42]


def patch_grid(h: int, w: int, p: int) -> list[tuple[int, int]]:
    """Top-left coords of non-overlapping p x p patches."""
    rows = list(range(0, h - p + 1, p))
    cols = list(range(0, w - p + 1, p))
    return [(r, c) for r in rows for c in cols]


def evaluate_video_qp_patch(video, qp, args, model, kind):
    PATCH = args.patch_size
    info = parse_info(args.orig_dir / f"{video}.y4m.info")
    width, height, fps = info["width"], info["height"], info["fps"]

    bitstream = args.bitstream_dir / f"{video}_QP{qp}.vvc"
    csv_path = args.decoded_dir / f"{video}_QP{qp}.csv"
    decoded_yuv = args.decoded_dir / f"{video}_QP{qp}_vtm_rec.yuv"
    orig_yuv = args.orig_dir / f"{video}.yuv"

    Y_o, U_o, V_o = read_yuv_planes(orig_yuv, width, height)
    Y_d, U_d, V_d = read_yuv_planes(decoded_yuv, width, height)
    n_frames = min(len(Y_o), len(Y_d))

    bitrate_kbps = bitstream.stat().st_size * 8 / 1000.0 * fps / n_frames

    tokens_by_poc = parse_csv_cached(csv_path)
    gen = FeatureMapGenerator(width, height)
    decoded_full = [to_full_res_chw(Y_d[i], U_d[i], V_d[i], height, width) for i in range(n_frames)]
    original_full = [to_full_res_chw(Y_o[i], U_o[i], V_o[i], height, width) for i in range(n_frames)]

    coords = patch_grid(height, width, PATCH)

    a_psnr_y = []; a_psnr_u = []; a_psnr_v = []
    e_psnr_y = []; e_psnr_u = []; e_psnr_v = []

    for i in range(1, n_frames - 1):
        feats_full = build_feature_stack(tokens_by_poc.get(i, []), gen)  # [9,H,W]
        for (r, c) in coords:
            sl = (slice(None), slice(r, r+PATCH), slice(c, c+PATCH))
            prev_p = decoded_full[i - 1][sl]
            curr_p = decoded_full[i][sl]
            next_p = decoded_full[i + 1][sl]
            orig_p = original_full[i][sl]
            feats_p = feats_full[sl]

            enhanced_p = run_model(model, kind, prev_p, curr_p, next_p, feats_p)

            # Convert both anchor and enhanced patches to native YUV420 uint8
            ya, ua, va = from_full_res_chw_to_planes(curr_p)
            ye, ue, ve = from_full_res_chw_to_planes(enhanced_p)
            yo, uo, vo = from_full_res_chw_to_planes(orig_p)

            a_psnr_y.append(psnr_uint8(yo, ya))
            a_psnr_u.append(psnr_uint8(uo, ua))
            a_psnr_v.append(psnr_uint8(vo, va))
            e_psnr_y.append(psnr_uint8(yo, ye))
            e_psnr_u.append(psnr_uint8(uo, ue))
            e_psnr_v.append(psnr_uint8(vo, ve))

    return {
        "qp": qp,
        "n_frames": n_frames,
        "n_patches": len(a_psnr_y),
        "fps": fps,
        "width": width,
        "height": height,
        "bitrate_kbps": bitrate_kbps,
        "psnr_anchor": {"Y": float(np.mean(a_psnr_y)), "U": float(np.mean(a_psnr_u)), "V": float(np.mean(a_psnr_v))},
        "psnr_enhanced": {"Y": float(np.mean(e_psnr_y)), "U": float(np.mean(e_psnr_u)), "V": float(np.mean(e_psnr_v))},
    }


def aggregate_and_bd(per_video, qps):
    agg = {qp: {"bitrate": [], "a": {"Y": [], "U": [], "V": []},
                 "e": {"Y": [], "U": [], "V": []}} for qp in qps}
    for vid, by_qp in per_video.items():
        for qp in qps:
            r = by_qp.get(qp)
            if r is None or "error" in r:
                continue
            agg[qp]["bitrate"].append(r["bitrate_kbps"])
            for ch in "YUV":
                agg[qp]["a"][ch].append(r["psnr_anchor"][ch])
                agg[qp]["e"][ch].append(r["psnr_enhanced"][ch])

    rd_mean = {qp: {} for qp in qps}
    for qp in qps:
        rd_mean[qp]["bitrate_kbps"] = float(np.mean(agg[qp]["bitrate"]))
        rd_mean[qp]["psnr_anchor"] = {ch: float(np.mean(agg[qp]["a"][ch])) for ch in "YUV"}
        rd_mean[qp]["psnr_enhanced"] = {ch: float(np.mean(agg[qp]["e"][ch])) for ch in "YUV"}

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


def main():
    from gpu_lock import acquire_gpu
    acquire_gpu("evaluate_patch")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["martell", "martell_nometa", "martell_unet", "snow_wide", "vvc_ppff", "stenet", "bi_conv_lstm", "qg_conv_lstm"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--qps", default="22,27,32,37,42")
    parser.add_argument("--orig-dir", type=Path, default=Path("data_eval"))
    parser.add_argument("--bitstream-dir", type=Path, default=Path("output_eval/encoded"))
    parser.add_argument("--decoded-dir", type=Path, default=Path("output_eval/decoded"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--videos", default=None)
    parser.add_argument("--patch-size", type=int, default=PATCH_DEFAULT,
                        help="non-overlapping patch size (px); default 132 to match training")
    args = parser.parse_args()

    qps = [int(q) for q in args.qps.split(",")]
    if args.videos:
        videos = args.videos.split(",")
    else:
        videos = sorted(p.stem for p in args.orig_dir.glob("*.yuv"))
    print(f"Patch eval ({args.patch_size}x{args.patch_size}). Videos ({len(videos)}): {videos}  QPs: {qps}")

    loaders = {"martell": load_martell, "martell_nometa": load_martell_nometa,
               "martell_unet": load_martell_unet,
               "snow_wide": load_snow_wide,
               "vvc_ppff": load_vvc_ppff, "stenet": load_stenet,
               "bi_conv_lstm": load_bi_conv_lstm,
               "qg_conv_lstm": load_qg_conv_lstm}
    model = loaders[args.model](args.checkpoint)

    per_video = {}
    pbar = tqdm(total=len(videos) * len(qps), desc=args.model)
    for video in videos:
        per_video[video] = {}
        for qp in qps:
            try:
                per_video[video][qp] = evaluate_video_qp_patch(video, qp, args, model, args.model)
                r = per_video[video][qp]
                pbar.set_postfix({"v": video[:12], "qp": qp,
                                  "ΔY": f"{r['psnr_enhanced']['Y']-r['psnr_anchor']['Y']:+.3f}"})
            except Exception as e:
                print(f"\n[ERROR] {video} QP{qp}: {e}")
                per_video[video][qp] = {"error": str(e)}
            pbar.update(1)
    pbar.close()

    summary = aggregate_and_bd(per_video, qps)
    out = {
        "model": args.model, "checkpoint": args.checkpoint, "qps": qps,
        "patch_size": args.patch_size, "videos": videos,
        "per_video": per_video, "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")
    print(json.dumps(summary["bd"], indent=2))


if __name__ == "__main__":
    main()
