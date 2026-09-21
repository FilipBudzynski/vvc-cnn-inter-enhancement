"""Fetch unbiased test videos for BD-rate evaluation."""

import argparse
import os
import subprocess
from pathlib import Path

import requests
from tqdm import tqdm

CHUNK_SIZE = 128 * 1024
BASE_URL = "https://media.xiph.org/video/derf/y4m/"
DEFAULT_FOLDER = "data_eval"
DEFAULT_FRAMES = 64

# content not in the training set: class E and 1080p class-B style
EVAL_VIDEOS = [
    "Johnny_1280x720_60.y4m",          # Class E talking head
    "vidyo1_720p_60fps.y4m",            # Class E news
    "vidyo3_720p_60fps.y4m",            # Class E news
    "controlled_burn_1080p.y4m",        # high-frequency texture
    "pedestrian_area_1080p25.y4m",      # JVET CTC, motion + people
    "sunflower_1080p25.y4m",            # static texture
    "tractor_1080p25.y4m",              # outdoor, fast motion
    "red_kayak_1080p.y4m",              # water + motion
    "rush_hour_1080p25.y4m",            # JVET CTC, traffic
    "touchdown_pass_1080p.y4m",         # sports
]


def process_video(target: Path, frames: int) -> None:
    dest = target.with_suffix(".yuv")
    info = target.with_name(target.name + ".info")

    subprocess.run(
        ["mediainfo", "-f", str(target)],
        stdout=open(info, "w"),
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(target),
            "-vf", "format=yuv420p",
            "-frames:v", str(frames),
            "-f", "rawvideo",
            str(dest),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    target.unlink()


def download(target_dir: Path, frames: int, force: bool = False) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)

    for video in EVAL_VIDEOS:
        y4m_path = target_dir / video
        yuv_path = y4m_path.with_suffix(".yuv")
        info_path = y4m_path.with_name(y4m_path.name + ".info")

        if yuv_path.exists() and info_path.exists() and not force:
            print(f"skip {video} (already extracted)")
            continue

        url = BASE_URL + video
        print(f"downloading {video} -> {y4m_path}")

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(y4m_path, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, leave=False
            ) as bar:
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
                    bar.update(len(chunk))

        print(f"converting {video}")
        process_video(y4m_path, frames)
        print(f"  -> {yuv_path} ({yuv_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", default=DEFAULT_FOLDER)
    parser.add_argument("-f", "--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    download(Path(args.dir), args.frames, args.force)
