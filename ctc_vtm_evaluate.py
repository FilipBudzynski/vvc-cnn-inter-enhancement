"""Multi-model full-frame evaluator for the VTM-encoded CTC bitstreams produced
by ctc_vtm_prepare.py (variants: in-loop filters "on" / "off").

Usage:
    uv run python ctc_vtm_evaluate.py --variant off
    uv run python ctc_vtm_evaluate.py --variant on --classes D,C
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

import math

import torch

import evaluate_bd as E
import ctc_evaluate as C
from ctc_bd import bd_for_qpsets

TILE_MAX_ROWS = 720
TILE_OVERLAP = 128


def _tile_rows_for_width(width: int) -> int:
    import os
    env = os.environ.get("VVC_TILE_MAX_ROWS")
    if env:  # per-run override for heavy model QG-ConvLSTM
        return max(128, int(env) if width <= 1920
                   else (int(env) * 1920) // width)
    """Cap tile AREA, not just rows: at 2560 px width a 720+2*128-row tile
    exceeds the 12 GB card and spills to WSL shared memory (~40x slowdown,
    observed 45 min/task on class A). Scale rows down so a tile never has
    more pixels than a 1920-wide one."""
    if width <= 1920:
        return TILE_MAX_ROWS
    return max(256, (TILE_MAX_ROWS * 1920) // width)


def run_model_tiled(model, kind, prev, curr, nxt, meta):
    """Full-frame below TILE_MAX_ROWS; otherwise N vertical tiles with
    TILE_OVERLAP extension rows on each side (seams discarded)."""
    H = curr.shape[1]
    max_rows = _tile_rows_for_width(curr.shape[2])
    if H <= max_rows:
        return E.run_model(model, kind, prev, curr, nxt, meta)
    import os as _os
    overlap = int(_os.environ.get("VVC_TILE_OVERLAP", TILE_OVERLAP))
    n = math.ceil(H / max_rows)
    bounds = [round(H * i / n) // 8 * 8 for i in range(1, n)]
    starts, ends = [0] + bounds, bounds + [H]
    out = torch.empty_like(curr)
    for s, e in zip(starts, ends):
        s_ext, e_ext = max(0, s - overlap), min(H, e + overlap)
        res = E.run_model(model, kind, prev[:, s_ext:e_ext], curr[:, s_ext:e_ext],
                          nxt[:, s_ext:e_ext], meta[:, s_ext:e_ext])
        out[:, s:e] = res[:, s - s_ext:(s - s_ext) + (e - s)].cpu()
        del res
    return out

QPSETS = {"qp4": [22, 27, 32, 37]}


def evaluate_video_qp(task, models):
    W, H, fps = task["width"], task["height"], task["fps"]
    qp = task["qp"]

    Y_o, U_o, V_o = E.read_yuv_planes(Path(task["input"]), W, H)
    Y_d, U_d, V_d = E.read_yuv_planes(Path(task["vtm_rec"]), W, H)
    n = min(len(Y_o), len(Y_d))
    Y_o, U_o, V_o, Y_d, U_d, V_d = (a[:n] for a in (Y_o, U_o, V_o, Y_d, U_d, V_d))

    size_bytes = Path(task["bitstream"]).stat().st_size
    bitrate_kbps = size_bytes * 8 / 1000.0 * fps / n

    anchor_frames = {ch: [E.psnr_uint8(o[i], d[i]) for i in range(n)]
                     for ch, o, d in (("Y", Y_o, Y_d), ("U", U_o, U_d), ("V", V_o, V_d))}

    toks = E.parse_csv_cached(Path(task["csv"]))
    gen = E.FeatureMapGenerator(W, H)
    dec_full = [E.to_full_res_chw(Y_d[i], U_d[i], V_d[i], H, W) for i in range(n)]

    enh = {m: {"Y": [], "U": [], "V": []} for m in models}
    t_model = {m: 0.0 for m in models}
    for i in range(1, n - 1):
        meta = E.build_feature_stack(toks.get(i, []), gen)
        for m, (model, kind) in models.items():
            t0 = time.time()
            out = run_model_tiled(model, kind, dec_full[i - 1], dec_full[i],
                                  dec_full[i + 1], meta)
            y_e, u_e, v_e = E.from_full_res_chw_to_planes(out)
            t_model[m] += time.time() - t0
            enh[m]["Y"].append(E.psnr_uint8(Y_o[i], y_e))
            enh[m]["U"].append(E.psnr_uint8(U_o[i], u_e))
            enh[m]["V"].append(E.psnr_uint8(V_o[i], v_e))

    interior = slice(1, n - 1)
    base = {
        "qp": qp, "n_frames": n, "n_enhanced": n - 2, "fps": fps,
        "width": W, "height": H, "cls": task["cls"],
        "train_contaminated": task["train_contaminated"],
        "bitrate_kbps": bitrate_kbps,
        "psnr_anchor": {ch: float(np.mean(v)) for ch, v in anchor_frames.items()},
        "psnr_anchor_interior": {ch: float(np.mean(v[interior]))
                                 for ch, v in anchor_frames.items()},
    }
    per_model = {}
    for m in models:
        per_model[m] = dict(base)
        per_model[m]["psnr_enhanced"] = {ch: float(np.mean(enh[m][ch])) for ch in "YUV"}
        per_model[m]["delta_psnr"] = {
            ch: per_model[m]["psnr_enhanced"][ch] - base["psnr_anchor_interior"][ch]
            for ch in "YUV"}
        per_model[m]["model_time_s"] = t_model[m]
    return per_model


def main():
    from gpu_lock import acquire_gpu
    acquire_gpu("ctc_vtm_evaluate")
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["on", "off"], required=True)
    ap.add_argument("--classes", default=None, help="e.g. D,C,E,B (default: all in manifest)")
    ap.add_argument("--models", default=",".join(C.MODELS))
    ap.add_argument("--out-dir", type=Path, default=Path("bdrate_results_ctc"))
    args = ap.parse_args()

    tasks, seen = [], set()
    manifests = sorted((Path("output_vtm") / args.variant)
                       .glob("prepare_manifest_*.json"))
    if not manifests:
        raise SystemExit(f"no prepare_manifest_*.json under output_vtm/{args.variant}")
    for mp in manifests:
        for t in json.load(open(mp))["tasks"]:
            k = (t["video"], t["qp"])
            if k not in seen:
                seen.add(k)
                tasks.append(t)
    if args.classes:
        keep = set(args.classes.split(","))
        tasks = [t for t in tasks if t["cls"] in keep]
    qps = sorted({t["qp"] for t in tasks})
    by_vq = {(t["video"], t["qp"]): t for t in tasks}
    videos = sorted({t["video"] for t in tasks})

    names = args.models.split(",")
    models = {}
    for nm in names:
        loader, kind, ckpt = C.MODELS[nm]
        models[nm] = (loader(ckpt), kind)
    print(f"[vtm-{args.variant}] videos={len(videos)} qps={qps} models={names} "
          f"device={E.DEVICE}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = args.out_dir / f"_partial_vtm_{args.variant}.json"
    partial = json.load(open(partial_path)) if partial_path.exists() else {}

    t_start = time.time()
    order = sorted(by_vq, key=lambda k: by_vq[k]["width"] * by_vq[k]["height"])
    pbar = tqdm(total=len(order), desc=f"vtm-{args.variant}")
    for (v, qp) in order:
        key = f"{v}|{qp}"
        have = partial.get(key, {})
        if all(m in have for m in names):
            pbar.update(1)
            continue
        t0 = time.time()
        res = evaluate_video_qp(by_vq[(v, qp)], models)
        have.update(res)
        have["_wall_s"] = time.time() - t0
        partial[key] = have
        json.dump(partial, open(partial_path, "w"))
        pbar.set_postfix({"v": v[:14], "qp": qp,
                          "dY": {m[:6]: f"{res[m]['delta_psnr']['Y']:+.3f}"
                                 for m in names}})
        pbar.update(1)
    pbar.close()

    for m in names:
        per_video = {}
        for key, have in partial.items():
            v, qp = key.split("|")
            if m in have and v in videos:
                per_video.setdefault(v, {})[qp] = have[m]
        cls_of = {t["video"]: t["cls"] for t in tasks}
        contaminated = {t["video"] for t in tasks if t["train_contaminated"]}
        # no train contaminated sequences
        clean = [v for v in per_video
                 if v not in contaminated and cls_of[v] != "F"]
        summary = {"all_clean": {"videos": clean,
                                 "per_qp": C.rd_curve(per_video, clean, qps)}}
        summary["all_clean"]["bd"] = bd_for_qpsets(summary["all_clean"]["per_qp"],
                                                   QPSETS)
        for cls in sorted({cls_of[v] for v in per_video}):
            vs = [v for v in clean if cls_of[v] == cls]
            if not vs:
                continue
            rd = C.rd_curve(per_video, vs, qps)
            summary[f"class_{cls}"] = {"videos": vs, "per_qp": rd,
                                       "bd": bd_for_qpsets(rd, QPSETS)}
        summary["per_video_bd"] = {v: bd_for_qpsets(C.rd_curve(per_video, [v], qps),
                                                    QPSETS)
                                   for v in per_video}
        out = {
            "model": m, "kind": C.MODELS[m][1], "checkpoint": C.MODELS[m][2],
            "variant": args.variant,
            "encoder": "VTM EncoderApp (encoder_randomaccess_vtm.cfg, GOP32, CRA, "
                       "InternalBitDepth 8), in-loop filters "
                       + ("ON (cfg defaults)" if args.variant == "on"
                          else "OFF (DeblockingFilterDisable=1, SAO=0, ALF=0, CCALF=0)"),
            "decoder": "VTM DecoderAnalyserApp, block-statistics trace",
            "qps": qps, "videos": videos,
            "tiling": {"max_rows": TILE_MAX_ROWS, "overlap_rows": TILE_OVERLAP},
            "train_contaminated": sorted(contaminated & set(per_video)),
            "eval_wall_s_this_run": time.time() - t_start,
            "per_video": per_video,
            "summary": summary,
        }
        p = args.out_dir / f"{m}_vtm_{args.variant}.json"
        json.dump(out, open(p, "w"), indent=1)
        bd = summary["all_clean"]["bd"]["qp4"]
        print(f"{m:22s} clean-avg  " + "  ".join(
            f"{ch} {bd[ch]['cubic']['bd_rate_pct']:+6.2f}%" for ch in "YUV")
            + f"  -> {p}")


if __name__ == "__main__":
    main()
