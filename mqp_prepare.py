"""Multi-QP training-data preparation for the corrective experiment
(separate from all existing data; nothing is overwritten):

Usage:
    uv run python mqp_prepare.py --variant off
    uv run python mqp_prepare.py --variant on --stage decode
"""

import argparse
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from ctc_vtm_prepare import encode_one, decode_one, parse_one, QPS, FRAMES

CANDIDATE_RES = [(176, 144), (352, 240), (352, 288), (704, 576),
                 (1280, 720), (1920, 1080)]


def guess_meta(path: Path):
    name = path.stem
    m = re.search(r"(\d{3,4})x(\d{3,4})", name)
    size = path.stat().st_size
    cands = [(int(m.group(1)), int(m.group(2)))] if m else CANDIDATE_RES
    res = None
    for w, h in cands:
        fb = w * h * 3 // 2
        if size % fb == 0 and size // fb >= FRAMES:
            res = (w, h)
            break
    if res is None:
        for w, h in CANDIDATE_RES:
            fb = w * h * 3 // 2
            if size % fb == 0 and size // fb >= FRAMES:
                res = (w, h)
                break
    assert res, f"cannot infer resolution of {path}"
    m = re.search(r"(?:^|[^0-9])(2[0-9]|[3-6][0-9])(?:fps|$|[^0-9x])", name)
    fps = None
    for pat in [r"720p(\d{2})", r"1080p(\d{2})", r"(\d{2})fps", r"_(\d{2})$"]:
        mm = re.search(pat, name)
        if mm:
            fps = int(mm.group(1))
            break
    if fps is None:
        fps = 60 if "5994" in name else 30
    return res[0], res[1], fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["on", "off"], required=True)
    ap.add_argument("--qps", default=",".join(map(str, QPS)))
    ap.add_argument("--stage", choices=["all", "encode", "decode", "parse"],
                    default="all")
    ap.add_argument("--enc-workers", type=int, default=13)
    ap.add_argument("--dec-workers", type=int, default=6)
    ap.add_argument("--parse-workers", type=int, default=4)
    args = ap.parse_args()
    qps = [int(q) for q in args.qps.split(",")]

    out = Path("output_vtm_train") / args.variant
    enc_dir, dec_dir = out / "encoded", out / "decoded"
    enc_dir.mkdir(parents=True, exist_ok=True)
    dec_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for yuv in sorted(Path("data").glob("*.yuv")):
        w, h, fps = guess_meta(yuv)
        for qp in qps:
            stem = yuv.stem
            tasks.append({
                "variant": args.variant, "cls": "train", "video": stem, "qp": qp,
                "width": w, "height": h, "fps": fps, "train_contaminated": False,
                "input": str(yuv),
                "bitstream": str(enc_dir / f"{stem}_QP{qp}.vvc"),
                "recon": str(enc_dir / f"{stem}_QP{qp}_rec.yuv"),
                "log": str(enc_dir / f"{stem}_QP{qp}.log"),
                "vtm_rec": str(dec_dir / f"{stem}_QP{qp}_vtm_rec.yuv"),
                "csv": str(dec_dir / f"{stem}_QP{qp}.csv"),
            })
    # sort big encodes first so the long pole starts early
    tasks.sort(key=lambda t: -(t["width"] * t["height"]))
    todo = [t for t in tasks
            if not (Path(t["bitstream"]).exists() and Path(t["recon"]).exists()
                    and Path(t["recon"]).stat().st_size
                    == t["width"] * t["height"] * 3 // 2 * FRAMES)]
    print(f"[mqp-{args.variant}] {len(tasks)} tasks, encode-todo={len(todo)}")

    stages = []
    if args.stage in ("all", "encode"):
        stages.append(("encode", encode_one, args.enc_workers, todo))
    if args.stage in ("all", "decode"):
        stages.append(("decode", decode_one, args.dec_workers, tasks))
    if args.stage in ("all", "parse"):
        stages.append(("parse", parse_one, args.parse_workers, tasks))

    stage_times = {}
    for stage, fn, workers, pool in stages:
        t0 = time.time()
        done = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fn, t): t for t in pool}
            for f in tqdm(as_completed(futs), total=len(futs),
                          desc=f"mqp-{args.variant}:{stage}"):
                done.append(f.result())
        stage_times[stage] = time.time() - t0
        print(f"[mqp-{args.variant}] {stage}: {stage_times[stage]/60:.1f} min wall",
              flush=True)
        if stage != "encode":
            tasks = done

    if args.stage in ("all", "decode"):
        bad = [t["video"] + f"_QP{t['qp']}" for t in tasks
               if not t.get("vtm_rec_equals_enc_rec", True)]
        print("VTM recon != encoder recon for:", bad if bad else "none")

    with open(out / f"manifest_{args.stage}.json", "w") as f:
        json.dump({"variant": args.variant, "qps": qps, "frames": FRAMES,
                   "stage_wall_s": stage_times, "tasks": tasks}, f, indent=2)


if __name__ == "__main__":
    main()
