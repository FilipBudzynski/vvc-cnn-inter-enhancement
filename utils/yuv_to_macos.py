#!/usr/bin/env python3
"""
YUV to macOS Video Format Conversion Utility

Converts YUV420 files to macOS-compatible video formats (MP4/MOV).
"""

import subprocess
import argparse
from pathlib import Path


def yuv_to_macos_video(input_file: str, output_file: str, width: int, height: int, fps: int = 30, format: str = "mp4"):
    """
    Convert YUV420 file to macOS-compatible video format using ffmpeg.
    
    Args:
        input_file: Input YUV file path
        output_file: Output video file path
        width: Video width
        height: Video height
        fps: Frame rate (default: 30)
        format: Output format (mp4 or mov)
    """
    cmd = [
        "ffmpeg",
        "-f", "rawvideo",
        "-pix_fmt", "yuv420p",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", input_file,
        "-c:v", "libx264",  # H.264 codec (macOS compatible)
        "-preset", "fast",
        "-crf", "23",  # Quality setting
        "-pix_fmt", "yuv420p",
        output_file
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Converted: {input_file} -> {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert YUV to macOS video format")
    parser.add_argument("input", help="Input YUV file")
    parser.add_argument("output", help="Output video file")
    parser.add_argument("--width", type=int, default=352, help="Video width")
    parser.add_argument("--height", type=int, default=288, help="Video height") 
    parser.add_argument("--fps", type=int, default=30, help="Frame rate")
    parser.add_argument("--format", choices=["mp4", "mov"], default="mp4", help="Output format")
    
    args = parser.parse_args()
    
    yuv_to_macos_video(args.input, args.output, args.width, args.height, args.fps, args.format)


if __name__ == "__main__":
    main()