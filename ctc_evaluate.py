"""Multi-model full-frame evaluator for the CTC-like (in-loop filters ON)
bitstreams produced by ctc_prepare.py.

Usage:
    uv run python ctc_evaluate.py --config ra
    uv run python ctc_evaluate.py --config ld
    uv run python ctc_evaluate.py --verify-tiling
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import evaluate_bd as E
from ctc_bd import bd_for_qpsets

QPSETS = {"qp4": [22, 27, 32, 37], "qp5": [22, 27, 32, 37, 42]}

MODELS = {
    # name           (loader,                kind,             checkpoint)
    "martell_hybrid":        (E.load_martell,        "martell",        "checkpoints/martell_hybrid_best.pt"),
    "martell_hybrid_nometa": (E.load_martell_nometa, "martell_nometa", "checkpoints/martell_hybrid_nometa_best.pt"),
    "martell_unet":          (E.load_martell_unet,   "martell_unet",   "checkpoints/martell_unet_best.pt"),
    "vvc_ppff":              (E.load_vvc_ppff,       "vvc_ppff",       "checkpoints/vvc_ppff_epoch_190.pt"),
    "stenet_2024":           (E.load_stenet,         "stenet",         "checkpoints/stenet_2024_fixed_best.pt"),
    "bi_conv_lstm":          (E.load_bi_conv_lstm,   "bi_conv_lstm",   "checkpoints/bi_conv_lstm_fixed_best.pt"),
    # multi-QP corrective experiment (trained on VTM-encoded data)
    "martell_hybrid_mqp_on":  (E.load_martell, "martell", "checkpoints/martell_hybrid_mqp_on_best.pt"),
    "martell_hybrid_mqp_off": (E.load_martell, "martell", "checkpoints/martell_hybrid_mqp_off_best.pt"),
    # mqpnwd = trained without weight decay
    "martell_hybrid_mqpnwd_on":  (E.load_martell, "martell", "checkpoints/martell_hybrid_mqpnwd_on_best.pt"),
    "martell_hybrid_mqpnwd_off": (E.load_martell, "martell", "checkpoints/martell_hybrid_mqpnwd_off_best.pt"),
    "martell_hybrid_abl0_off": (E.load_martell, "martell", "checkpoints/martell_hybrid_abl0_off_best.pt"),  # bez QP
    "martell_hybrid_abl1_off": (E.load_martell, "martell", "checkpoints/martell_hybrid_abl1_off_best.pt"),  # bez PredMode
    "martell_hybrid_abl2_off": (E.load_martell, "martell", "checkpoints/martell_hybrid_abl2_off_best.pt"),  # bez Depth
    "martell_hybrid_abl3_off": (E.load_martell, "martell", "checkpoints/martell_hybrid_abl3_off_best.pt"),  # bez Boundary
    "martell_hybrid_abl4_off": (E.load_martell, "martell", "checkpoints/martell_hybrid_abl4_off_best.pt"),  # bez MVL0_X
    "martell_hybrid_abl5_off": (E.load_martell, "martell", "checkpoints/martell_hybrid_abl5_off_best.pt"),  # bez MVL0_Y
    "martell_hybrid_abl6_off": (E.load_martell, "martell", "checkpoints/martell_hybrid_abl6_off_best.pt"),  # bez MVL1_X
    "martell_hybrid_abl7_off": (E.load_martell, "martell", "checkpoints/martell_hybrid_abl7_off_best.pt"),  # bez MVL1_Y
    "martell_hybrid_abl8_off": (E.load_martell, "martell", "checkpoints/martell_hybrid_abl8_off_best.pt"),  # bez FrameType
    "martell_hybrid_abl_allMV_off": (E.load_martell, "martell", "checkpoints/martell_hybrid_abl_allMV_off_best.pt"),  # bez 4 kanalow MV
    "martell_hybrid_nometa_mqp_on":  (E.load_martell_nometa, "martell_nometa", "checkpoints/martell_hybrid_nometa_mqp_on_best.pt"),
    "martell_hybrid_nometa_mqp_off": (E.load_martell_nometa, "martell_nometa", "checkpoints/martell_hybrid_nometa_mqp_off_best.pt"),
    "vvc_ppff_mqp_on":  (E.load_vvc_ppff, "vvc_ppff", "checkpoints/vvc_ppff_mqp_on_best.pt"),
    "vvc_ppff_mqp_off": (E.load_vvc_ppff, "vvc_ppff", "checkpoints/vvc_ppff_mqp_off_best.pt"),
    # mqp2 = repeat with late LR milestones [30,40,46,50] (slow-converging net)
    "vvc_ppff_mqp2_on":  (E.load_vvc_ppff, "vvc_ppff", "checkpoints/vvc_ppff_mqp2_on_best.pt"),
    "vvc_ppff_mqp2_off": (E.load_vvc_ppff, "vvc_ppff", "checkpoints/vvc_ppff_mqp2_off_best.pt"),
    # mqp3 = faithful original procedure: clamp(0,1) before the loss, wd=1e-4
    "vvc_ppff_mqp3_on":  (E.load_vvc_ppff, "vvc_ppff", "checkpoints/vvc_ppff_mqp3_on_best.pt"),
    "vvc_ppff_mqp3_off": (E.load_vvc_ppff, "vvc_ppff", "checkpoints/vvc_ppff_mqp3_off_best.pt"),
    "stenet_2024_mqp_on":  (E.load_stenet, "stenet", "checkpoints/stenet_2024_mqp_on_best.pt"),
    "stenet_2024_mqp_off": (E.load_stenet, "stenet", "checkpoints/stenet_2024_mqp_off_best.pt"),
    "bi_conv_lstm_mqp_on":  (E.load_bi_conv_lstm, "bi_conv_lstm", "checkpoints/bi_conv_lstm_mqp_on_best.pt"),
    "bi_conv_lstm_mqp_off": (E.load_bi_conv_lstm, "bi_conv_lstm", "checkpoints/bi_conv_lstm_mqp_off_best.pt"),
    "qg_conv_lstm_mqp_on":  (E.load_qg_conv_lstm, "qg_conv_lstm", "checkpoints/qg_conv_lstm_mqp_on_best.pt"),
    "qg_conv_lstm_mqp_off": (E.load_qg_conv_lstm, "qg_conv_lstm", "checkpoints/qg_conv_lstm_mqp_off_best.pt"),
}

TILE_MAX_ROWS = 720
TILE_OVERLAP = 256


def run_model_tiled(model, kind, prev, curr, nxt, meta):
    """Full-frame for <=720 rows; otherwise two vertical tiles with overlap."""
    H = curr.shape[1]
    if H <= TILE_MAX_ROWS:
        return E.run_model(model, kind, prev, curr, nxt, meta)
    mid = H // 2
    ov = TILE_OVERLAP
    sl_top, sl_bot = slice(0, mid + ov), slice(mid - ov, H)
    out = torch.empty_like(curr)
    top = E.run_model(model, kind, prev[:, sl_top], curr[:, sl_top], nxt[:, sl_top], meta[:, sl_top])
    out[:, :mid] = top[:, :mid].cpu()
    del top
    bot = E.run_model(model, kind, prev[:, sl_bot], curr[:, sl_bot], nxt[:, sl_bot], meta[:, sl_bot])
    out[:, mid:] = bot[:, ov:].cpu()
    del bot
    return out


def verify_tiling(models, args):
    """Compare tiled vs full-frame output on real 720p frames (fits in memory)."""
    video, qp = "Johnny_1280x720_60", 32
    dec_dir = Path("output_ctc") / args.config / "decoded"
    orig_dir = Path("data_eval")
    info = E.parse_info(orig_dir / f"{video}.y4m.info")
    W, H = info["width"], info["height"]
    Y_d, U_d, V_d = E.read_yuv_planes(dec_dir / f"{video}_QP{qp}_vtm_rec.yuv", W, H)
    toks = E.parse_csv_cached(dec_dir / f"{video}_QP{qp}.csv")
    gen = E.FeatureMapGenerator(W, H)
    frames = [E.to_full_res_chw(Y_d[i], U_d[i], V_d[i], H, W) for i in range(0, 8)]
    global TILE_MAX_ROWS
    saved = TILE_MAX_ROWS
    for name, (model, kind) in models.items():
        worst = 0.0
        worst_u8 = 0
        for i in range(1, 7):
            meta = E.build_feature_stack(toks.get(i, []), gen)
            TILE_MAX_ROWS = 10_000
            full = E.run_model(model, kind, frames[i - 1], frames[i], frames[i + 1], meta).cpu()
            TILE_MAX_ROWS = 0
            tiled = run_model_tiled(model, kind, frames[i - 1], frames[i], frames[i + 1], meta).cpu()
            worst = max(worst, float((full - tiled).abs().max()))
            f8 = (full.clamp(0, 1) * 255).round()
            t8 = (tiled.clamp(0, 1) * 255).round()
            worst_u8 = max(worst_u8, int((f8 != t8).sum()))
        print(f"[verify-tiling] {name:22s} max|full-tiled| = {worst:.2e} (float, [0,1]); "
              f"differing 8-bit samples over 6 frames: {worst_u8}")
    TILE_MAX_ROWS = saved


def evaluate_video_qp(video, qp, cfg_dirs, models):
    orig_dir, enc_dir, dec_dir = cfg_dirs
    info = E.parse_info(orig_dir / f"{video}.y4m.info")
    W, H, fps = info["width"], info["height"], info["fps"]

    Y_o, U_o, V_o = E.read_yuv_planes(orig_dir / f"{video}.yuv", W, H)
    Y_d, U_d, V_d = E.read_yuv_planes(dec_dir / f"{video}_QP{qp}_vtm_rec.yuv", W, H)
    n = min(len(Y_o), len(Y_d))
    Y_o, U_o, V_o, Y_d, U_d, V_d = (a[:n] for a in (Y_o, U_o, V_o, Y_d, U_d, V_d))

    size_bytes = (enc_dir / f"{video}_QP{qp}.vvc").stat().st_size
    bitrate_kbps = size_bytes * 8 / 1000.0 * fps / n

    anchor_frames = {ch: [E.psnr_uint8(o[i], d[i]) for i in range(n)]
                     for ch, o, d in (("Y", Y_o, Y_d), ("U", U_o, U_d), ("V", V_o, V_d))}

    toks = E.parse_csv_cached(dec_dir / f"{video}_QP{qp}.csv")
    gen = E.FeatureMapGenerator(W, H)
    dec_full = [E.to_full_res_chw(Y_d[i], U_d[i], V_d[i], H, W) for i in range(n)]

    enh = {m: {"Y": [], "U": [], "V": []} for m in models}
    t_model = {m: 0.0 for m in models}
    for i in range(1, n - 1):
        meta = E.build_feature_stack(toks.get(i, []), gen)
        for m, (model, kind) in models.items():
            t0 = time.time()
            out = run_model_tiled(model, kind, dec_full[i - 1], dec_full[i], dec_full[i + 1], meta)
            y_e, u_e, v_e = E.from_full_res_chw_to_planes(out)
            t_model[m] += time.time() - t0
            enh[m]["Y"].append(E.psnr_uint8(Y_o[i], y_e))
            enh[m]["U"].append(E.psnr_uint8(U_o[i], u_e))
            enh[m]["V"].append(E.psnr_uint8(V_o[i], v_e))

    interior = slice(1, n - 1)
    base = {
        "qp": qp, "n_frames": n, "n_enhanced": n - 2, "fps": fps, "width": W, "height": H,
        "bitrate_kbps": bitrate_kbps,
        "psnr_anchor": {ch: float(np.mean(v)) for ch, v in anchor_frames.items()},
        "psnr_anchor_interior": {ch: float(np.mean(v[interior])) for ch, v in anchor_frames.items()},
        "psnr_anchor_per_frame": {ch: [round(x, 4) for x in v] for ch, v in anchor_frames.items()},
    }
    per_model = {}
    for m in models:
        per_model[m] = dict(base)
        per_model[m]["psnr_enhanced"] = {ch: float(np.mean(enh[m][ch])) for ch in "YUV"}
        per_model[m]["delta_psnr"] = {ch: per_model[m]["psnr_enhanced"][ch] - base["psnr_anchor_interior"][ch]
                                      for ch in "YUV"}
        per_model[m]["psnr_enhanced_per_frame"] = {ch: [round(x, 4) for x in enh[m][ch]] for ch in "YUV"}
        per_model[m]["model_time_s"] = t_model[m]
    return per_model


def rd_curve(per_video: dict, videos: list, qps: list) -> dict:
    """Average (rate, PSNR) over the given videos per QP (as evaluate_bd.aggregate_and_bd)."""
    rd = {}
    for qp in qps:
        rows = [per_video[v][str(qp)] for v in videos if str(qp) in per_video[v]]
        if len(rows) != len(videos):
            continue
        rd[qp] = {
            "bitrate_kbps": float(np.mean([r["bitrate_kbps"] for r in rows])),
            "psnr_anchor": {ch: float(np.mean([r["psnr_anchor_interior"][ch] for r in rows])) for ch in "YUV"},
            "psnr_enhanced": {ch: float(np.mean([r["psnr_enhanced"][ch] for r in rows])) for ch in "YUV"},
        }
    return rd


def summarize_model(per_video: dict, qps: list) -> dict:
    videos = sorted(per_video)
    summary = {"all_videos": {"videos": videos, "per_qp": rd_curve(per_video, videos, qps)}}
    summary["all_videos"]["bd"] = bd_for_qpsets(summary["all_videos"]["per_qp"], QPSETS)
    summary["per_video_bd"] = {}
    for v in videos:
        rd = rd_curve(per_video, [v], qps)
        summary["per_video_bd"][v] = bd_for_qpsets(rd, QPSETS)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["ra", "ld"], default="ra")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--qps", default="22,27,32,37,42")
    ap.add_argument("--videos", default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("bdrate_results_ctc"))
    ap.add_argument("--verify-tiling", action="store_true")
    args = ap.parse_args()

    qps = [int(q) for q in args.qps.split(",")]
    manifest = json.load(open(Path("output_ctc") / args.config / "prepare_manifest.json"))
    src_dir = {t["video"]: Path(t["input"]).parent for t in manifest["tasks"]}
    videos = args.videos.split(",") if args.videos else sorted(src_dir)
    enc_dir = Path("output_ctc") / args.config / "encoded"
    dec_dir = Path("output_ctc") / args.config / "decoded"

    names = args.models.split(",")
    models = {}
    for nm in names:
        loader, kind, ckpt = MODELS[nm]
        models[nm] = (loader(ckpt), kind)
    print(f"[{args.config}] videos={videos} qps={qps} models={names} device={E.DEVICE}")

    if args.verify_tiling:
        verify_tiling(models, args)
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = args.out_dir / f"_partial_{args.config}.json"
    partial = json.load(open(partial_path)) if partial_path.exists() else {}

    t_start = time.time()
    pbar = tqdm(total=len(videos) * len(qps), desc=f"ctc-{args.config}")
    for v in videos:
        for qp in qps:
            key = f"{v}|{qp}"
            have = partial.get(key, {})
            if all(m in have for m in names):
                pbar.update(1)
                continue
            t0 = time.time()
            res = evaluate_video_qp(v, qp, (src_dir[v], enc_dir, dec_dir), models)
            have.update(res)
            have["_wall_s"] = time.time() - t0
            partial[key] = have
            json.dump(partial, open(partial_path, "w"))
            pbar.set_postfix({"v": v[:14], "qp": qp,
                              "dY": {m[:6]: f"{res[m]['delta_psnr']['Y']:+.3f}" for m in names},
                              "s": f"{have['_wall_s']:.0f}"})
            pbar.update(1)
    pbar.close()
    eval_wall = time.time() - t_start

    for m in names:
        per_video = {}
        for key, have in partial.items():
            v, qp = key.split("|")
            if m in have:
                per_video.setdefault(v, {})[qp] = have[m]
        per_video = {v: per_video[v] for v in videos if v in per_video}
        out = {
            "model": m, "kind": MODELS[m][1], "checkpoint": MODELS[m][2],
            "config": args.config,
            "encoder": "VVenC 1.15.0-dev, preset fast, in-loop filters ON (deblocking, SAO, ALF, CCALF)"
                       + (", Low Delay B (lowdelay_fast.cfg, GOP8, intra period -1)" if args.config == "ld"
                          else ", Random Access (GOP32, CRA, intra period 1 s)"),
            "decoder": "VTM DecoderAnalyserApp, D_BLOCK_STATISTICS_ALL trace",
            "qps": qps, "videos": list(per_video),
            "tiling": {"max_rows_full_frame": TILE_MAX_ROWS, "overlap_rows": TILE_OVERLAP},
            "eval_wall_s_this_run": eval_wall,
            "per_video": per_video,
            "summary": summarize_model(per_video, qps),
        }
        p = args.out_dir / f"{m}_{args.config}.json"
        json.dump(out, open(p, "w"), indent=1)
        bd = out["summary"]["all_videos"]["bd"]
        print(f"{m:22s} " + "  ".join(
            f"{qs}:{ch} {bd[qs][ch]['cubic']['bd_rate_pct']:+6.2f}%/{bd[qs][ch]['pchip']['bd_rate_pct']:+6.2f}%"
            for qs in bd for ch in "YUV"))
        print(f"  -> {p}")


if __name__ == "__main__":
    main()
