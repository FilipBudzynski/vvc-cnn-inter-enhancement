"""CTC-like preparation: encode with VVenC (preset fast, 64 frames) with the
in-loop filters ENABLED (deblocking + SAO + ALF/CCALF, i.e. VVenC defaults),
then decode with the VTM analyser (block-statistics trace on) exactly like
prepare_eval.py / prepare_jvet.py, and pre-parse the traces to .pkl.

Usage:
    uv run python ctc_prepare.py --config ra
    uv run python ctc_prepare.py --config ld
"""

import argparse
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from decoder.config import DecodingTaskParams
from decoder.decoders import VTMDecoder
from evaluate_bd import parse_csv_cached, parse_info

VVENC = "./bin/vvenc/bin/release-static/vvencFFapp"
LD_CFG = "bin/vvenc/cfg/experimental/lowdelay_fast.cfg"
QPS = [22, 27, 32, 37, 42]
FRAMES = 64

RA_SEQUENCES = {
    "data_eval": [
        "Johnny_1280x720_60", "vidyo1_720p_60fps", "vidyo3_720p_60fps",
        "controlled_burn_1080p", "pedestrian_area_1080p25", "sunflower_1080p25",
        "tractor_1080p25", "red_kayak_1080p", "rush_hour_1080p25",
        "touchdown_pass_1080p",
    ],
    "data_jvet": ["FourPeople_1280x720_60", "KristenAndSara_1280x720_60"],
}
LD_SEQUENCES = {
    "data_jvet": ["Johnny_1280x720_60", "FourPeople_1280x720_60",
                  "KristenAndSara_1280x720_60"],
}


def encode_one(task: dict) -> dict:
    cmd = [VVENC, "--preset", "fast"]
    if task["config"] == "ld":
        cmd += ["-c", LD_CFG]
    cmd += [
        "-i", task["input"],
        "-s", f"{task['width']}x{task['height']}",
        "-fr", str(task["fps"]),
        "-f", str(FRAMES),
        "-q", str(task["qp"]),
        "-b", task["bitstream"],
        "-o", task["recon"],
        "--InputChromaFormat", "420",
        "--ChromaFormatIDC", "420",
        "--InternalBitDepth", "8",
        "--OutputBitDepth", "8",
    ]
    t0 = time.time()
    with open(task["log"], "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.PIPE, text=True, check=True)
    task["encode_s"] = time.time() - t0
    task["cmd"] = " ".join(cmd)
    return task


def decode_one(task: dict) -> dict:
    dec = VTMDecoder()
    t0 = time.time()
    dec.decode(DecodingTaskParams(
        bitstream_input=task["bitstream"],
        output_yuv=task["vtm_rec"],
        trace_file=task["csv"],
    ))
    task["decode_s"] = time.time() - t0
    # VTM reconstruction must equal the encoder reconstruction
    same = subprocess.run(["cmp", "-s", task["vtm_rec"], task["recon"]]).returncode == 0
    task["vtm_rec_equals_vvenc_rec"] = same
    return task


def parse_one(task: dict) -> dict:
    t0 = time.time()
    toks = parse_csv_cached(Path(task["csv"]))
    task["parse_s"] = time.time() - t0
    task["n_pocs"] = len(toks)
    return task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["ra", "ld"], required=True)
    ap.add_argument("--qps", default=",".join(map(str, QPS)))
    ap.add_argument("--enc-workers", type=int, default=8)
    ap.add_argument("--dec-workers", type=int, default=6)
    ap.add_argument("--parse-workers", type=int, default=4)
    args = ap.parse_args()
    qps = [int(q) for q in args.qps.split(",")]

    seqs = RA_SEQUENCES if args.config == "ra" else LD_SEQUENCES
    out = Path("output_ctc") / args.config
    enc_dir, dec_dir = out / "encoded", out / "decoded"
    enc_dir.mkdir(parents=True, exist_ok=True)
    dec_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for data_dir, stems in seqs.items():
        for stem in stems:
            info = parse_info(Path(data_dir) / f"{stem}.y4m.info")
            for qp in qps:
                tasks.append({
                    "config": args.config, "video": stem, "qp": qp,
                    "input": str(Path(data_dir) / f"{stem}.yuv"),
                    "width": info["width"], "height": info["height"],
                    "fps": int(round(info["fps"])),
                    "bitstream": str(enc_dir / f"{stem}_QP{qp}.vvc"),
                    "recon": str(enc_dir / f"{stem}_QP{qp}_rec.yuv"),
                    "log": str(enc_dir / f"{stem}_QP{qp}.log"),
                    "vtm_rec": str(dec_dir / f"{stem}_QP{qp}_vtm_rec.yuv"),
                    "csv": str(dec_dir / f"{stem}_QP{qp}.csv"),
                })
    print(f"[{args.config}] {len(tasks)} (video, QP) tasks")

    stage_times = {}
    for stage, fn, workers in [("encode", encode_one, args.enc_workers),
                               ("decode", decode_one, args.dec_workers),
                               ("parse", parse_one, args.parse_workers)]:
        t0 = time.time()
        done = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fn, t): t for t in tasks}
            for f in tqdm(as_completed(futs), total=len(futs), desc=f"{args.config}:{stage}"):
                done.append(f.result())
        tasks = done
        stage_times[stage] = time.time() - t0
        print(f"[{args.config}] {stage}: {stage_times[stage]/60:.1f} min wall")

    bad = [t["video"] + f"_QP{t['qp']}" for t in tasks if not t["vtm_rec_equals_vvenc_rec"]]
    print("VTM recon != VVenC recon for:", bad if bad else "none (all identical)")

    with open(out / "prepare_manifest.json", "w") as f:
        json.dump({"config": args.config, "qps": qps, "frames": FRAMES,
                   "stage_wall_s": stage_times, "tasks": tasks}, f, indent=2)


if __name__ == "__main__":
    main()
