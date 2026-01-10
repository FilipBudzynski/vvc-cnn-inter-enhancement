from tqdm import tqdm
import os
import requests
import argparse
import subprocess


CHUNK_SIZE = 128 * 1024  # 128 KB
DEFAULT_FOLDER = "data"
BASE_URL = "https://media.xiph.org/video/derf/y4m/"

SELECTED_VIDEOS = [
    "crew_cif.y4m",
]


def process_video(target: str) -> None:
    parts = target.split(".")
    parts[-1] = "yuv"
    dest = ".".join(parts)
    ffmpeg_cmd = (
        f"ffmpeg -y -i {target} -c:v rawvideo -pixel_format yuv420p -frames:v 64 {dest}"
    )
    mediainfo_cmd = f"mediainfo -f {target} > {target}.info"
    subprocess.call(mediainfo_cmd, shell=True)
    subprocess.call(ffmpeg_cmd, shell=True)
    os.remove(target)


def download_videos(target: str) -> None:
    for video in tqdm(SELECTED_VIDEOS):
        url = os.path.join(BASE_URL, video)
        print(f"downloading {video}...")
        target_filename = os.path.join(target, video)

        if os.path.exists(target_filename):
            should_proceed = (
                input(f"file {target_filename} already exists, overwrite it? (y/n)")
                .strip()
                .upper()
            )
            if should_proceed != "Y":
                continue

        with requests.get(url, stream=True) as data:
            data.raise_for_status()

            with open(target_filename, "wb") as f:
                for chunk in tqdm(
                    data.iter_content(chunk_size=CHUNK_SIZE), leave=False
                ):
                    f.write(chunk)

        process_video(target_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", action="store", default=DEFAULT_FOLDER)

    args = parser.parse_args()

    if not os.path.exists(args.dir):
        os.mkdir(args.dir)

    download_videos(args.dir)
