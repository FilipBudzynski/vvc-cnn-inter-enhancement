#!/usr/bin/env python3
"""
YUV to Y4M Conversion Utility

Converts YUV420 files to Y4M format for playback compatibility.
"""

import subprocess
import argparse
from pathlib import Path


def yuv_to_y4m(input_file: str, output_file: str, width: int, height: int, fps: int = 30):
    """
    Convert YUV420 file to Y4M format using ffmpeg.
    
    Args:
        input_file: Input YUV file path
        output_file: Output Y4M file path  
        width: Video width
        height: Video height
        fps: Frame rate (default: 30)
    """
    cmd = [
        "ffmpeg",
        "-f", "rawvideo",
        "-pix_fmt", "yuv420p",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", input_file,
        "-f", "yuv4mpegpipe",
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
    parser = argparse.ArgumentParser(description="Convert YUV to Y4M format")
    parser.add_argument("input", help="Input YUV file")
    parser.add_argument("output", help="Output Y4M file")
    parser.add_argument("--width", type=int, default=352, help="Video width")
    parser.add_argument("--height", type=int, default=288, help="Video height") 
    parser.add_argument("--fps", type=int, default=30, help="Frame rate")
    
    args = parser.parse_args()
    
    yuv_to_y4m(args.input, args.output, args.width, args.height, args.fps)


if __name__ == "__main__":
    main()