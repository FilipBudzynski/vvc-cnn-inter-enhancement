import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    """Base configuration for the decoder."""

    bitstream_path: List[str] = field(default_factory=list)
    max_workers: Optional[int] = os.cpu_count()


BASE_CONFIG = Config()


@dataclass
class DecodingTaskParams:
    """Represents a single decoding job."""

    bitstream_path: str = "./output/encoded"
    output_yuv: str = "./output/decoded"
    trace_file: str = "./output/decoded"
