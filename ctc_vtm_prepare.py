"""VTM-based CTC preparation (promotor's methodology):

Usage:
    uv run python ctc_vtm_prepare.py --variant off --classes D,C,E,B
    uv run python ctc_vtm_prepare.py --variant on  --classes D,C,E,B
    uv run python ctc_vtm_prepare.py --variant off --stage decode
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
from evaluate_bd import parse_csv_cached

ENCODER = "bin/vtm/bin/EncoderAppStatic"
CFG = "bin/vtm/cfg/encoder_randomaccess_vtm.cfg"
QPS = [22, 27, 32, 37]
FRAMES = 64

# (file stem, width, height, fps); trained-pool contamination flags below
SEQUENCES = {
    "A": [
        ("NebutaFestival_60fps_f300", 2560, 1600, 60),
        ("PeopleOnStreet_2560x1600_30_crop", 2560, 1600, 30),
        ("SteamLocomotiveTrain_60fps_f300", 2560, 1600, 60),
        ("Traffic_2560x1600_30_crop", 2560, 1600, 30),
    ],
    "B": [
        ("BasketballDrive_1920x1080_50", 1920, 1080, 50),
        ("BQTerrace_1920x1080_60", 1920, 1080, 60),
        ("Cactus_1920x1080_50", 1920, 1080, 50),
        ("Kimono1_1920x1080_24", 1920, 1080, 24),
        ("ParkScene_1920x1080_24", 1920, 1080, 24),
    ],
    "C": [
        ("BasketballDrill_832x480_50", 832, 480, 50),
        ("BQMall_832x480_60", 832, 480, 60),
        ("PartyScene_832x480_50", 832, 480, 50),
        ("RaceHorses_832x480_30", 832, 480, 30),
    ],
    "D": [
        ("BasketballPass_416x240_50", 416, 240, 50),
        ("BlowingBubbles_416x240_50", 416, 240, 50),
        ("BQSquare_416x240_60", 416, 240, 60),
        ("RaceHorses_416x240_30", 416, 240, 30),
    ],
    "F": [
        ("BasketballDrillText_832x480_50", 832, 480, 50),
        ("ChinaSpeed_1024x768_30", 1024, 768, 30),
        ("SlideEditing_1280x720_30", 1280, 720, 30),
        ("SlideShow_1280x720_20", 1280, 720, 20),
    ],
    "E": [
        ("FourPeople_1280x720_60", 1280, 720, 60),
        ("Johnny_1280x720_60", 1280, 720, 60),
        ("KristenAndSara_1280x720_60", 1280, 720, 60),
        ("Vidyo1_60fps_f600", 1280, 720, 60),
        ("Vidyo3_60fps_f600", 1280, 720, 60),
        ("Vidyo4_60fps_f600", 1280, 720, 60),
    ],
}

TRAIN_CONTAMINATED = {"FourPeople_1280x720_60", "KristenAndSara_1280x720_60"}


def encode_one(task: dict) -> dict:
    cmd = [
        ENCODER, "-c", CFG,
        "-i", task["input"],
        "-b", task["bitstream"],
        "-o", task["recon"],
        "-wdt", str(task["width"]),
        "-hgt", str(task["height"]),
        "-fr", str(task["fps"]),
        "-f", str(FRAMES),
        "-q", str(task["qp"]),
        "--InputBitDepth=8",
        "--InternalBitDepth=8",
        "--OutputBitDepth=8",
        "--InputChromaFormat=420",
        "--Level=5.1",
    ]
    if task["variant"] == "off":
        cmd += ["--DeblockingFilterDisable=1", "--SAO=0", "--ALF=0", "--CCALF=0"]
    t0 = time.time()
    with open(task["log"], "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, check=True)
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
    same = subprocess.run(["cmp", "-s", task["vtm_rec"], task["recon"]]).returncode == 0
    task["vtm_rec_equals_enc_rec"] = same
    return task


def parse_one(task: dict) -> dict:
    t0 = time.time()
    toks = parse_csv_cached(Path(task["csv"]))
    task["parse_s"] = time.time() - t0
    task["n_pocs"] = len(toks)
    return task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["on", "off"], required=True)
    ap.add_argument("--classes", default="D,C,E,B")
    ap.add_argument("--qps", default=",".join(map(str, QPS)))
    ap.add_argument("--stage", choices=["all", "encode", "decode", "parse"],
                    default="all")
    ap.add_argument("--enc-workers", type=int, default=13)
    ap.add_argument("--dec-workers", type=int, default=6)
    ap.add_argument("--parse-workers", type=int, default=4)
    args = ap.parse_args()
    qps = [int(q) for q in args.qps.split(",")]
    classes = args.classes.split(",")

    out = Path("output_vtm") / args.variant
    enc_dir, dec_dir = out / "encoded", out / "decoded"
    enc_dir.mkdir(parents=True, exist_ok=True)
    dec_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for cls in classes:
        for stem, w, h, fps in SEQUENCES[cls]:
            for qp in qps:
                tasks.append({
                    "variant": args.variant, "cls": cls, "video": stem, "qp": qp,
                    "width": w, "height": h, "fps": fps,
                    "train_contaminated": stem in TRAIN_CONTAMINATED,
                    "input": str(Path("data_hevc/hevc") / cls / f"{stem}.yuv"),
                    "bitstream": str(enc_dir / f"{stem}_QP{qp}.vvc"),
                    "recon": str(enc_dir / f"{stem}_QP{qp}_rec.yuv"),
                    "log": str(enc_dir / f"{stem}_QP{qp}.log"),
                    "vtm_rec": str(dec_dir / f"{stem}_QP{qp}_vtm_rec.yuv"),
                    "csv": str(dec_dir / f"{stem}_QP{qp}.csv"),
                })
    # skip already-finished encodes (resume support)
    if args.stage in ("all", "encode"):
        todo = [t for t in tasks
                if not (Path(t["bitstream"]).exists()
                        and Path(t["recon"]).exists()
                        and Path(t["recon"]).stat().st_size
                        == t["width"] * t["height"] * 3 // 2 * FRAMES)]
    print(f"[{args.variant}] {len(tasks)} tasks, encode-todo="
          f"{len(todo) if args.stage in ('all', 'encode') else '-'}")

    stages = []
    if args.stage in ("all", "encode"):
        stages.append(("encode", encode_one, args.enc_workers, todo))
    if args.stage in ("all", "decode"):
        stages.append(("decode", decode_one, args.dec_workers, tasks))
    if args.stage in ("all", "parse"):
        stages.append(("parse", parse_one, args.parse_workers, tasks))

    stage_times = {}
    results = {(-1,): None}
    for stage, fn, workers, pool in stages:
        t0 = time.time()
        done = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fn, t): t for t in pool}
            for f in tqdm(as_completed(futs), total=len(futs),
                          desc=f"{args.variant}:{stage}"):
                r = f.result()
                done.append(r)
                if stage == "encode":
                    print(f"  enc {r['video']} QP{r['qp']}: {r['encode_s']/60:.1f} min",
                          flush=True)
        stage_times[stage] = time.time() - t0
        print(f"[{args.variant}] {stage}: {stage_times[stage]/60:.1f} min wall")
        if stage != "encode":
            tasks = done

    if args.stage in ("all", "decode"):
        bad = [t["video"] + f"_QP{t['qp']}" for t in tasks
               if not t.get("vtm_rec_equals_enc_rec", True)]
        print("VTM recon != encoder recon for:", bad if bad else "none")

    with open(out / f"prepare_manifest_{args.stage}_{'-'.join(classes)}.json", "w") as f:
        json.dump({"variant": args.variant, "qps": qps, "frames": FRAMES,
                   "classes": classes, "stage_wall_s": stage_times,
                   "tasks": tasks}, f, indent=2)


if __name__ == "__main__":
    main()
