import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    """Base configuration for the encoder."""

    data_dir: str = "./data"
    output_dir: str = "./encoded_output"
    encoder_path: str = "./bin/vvencFFapp"
    qp: List[int] = field(default_factory=lambda: [23])
    frames_to_encode: int = 64
    preset: str = "fast"
    alf: int = 1
    sao: int = 1
    max_workers: Optional[int] = os.cpu_count()


BASE_CONFIG = Config()


@dataclass
class EncodingTaskParams:
    """Represents a single encoding job."""

    input_file: str
    width: int
    height: int
    fps: int
    frames: int
    qp: int
    bitstream_out: str
    recon_out: str
    preset: str
    alf: int
    sao: int
