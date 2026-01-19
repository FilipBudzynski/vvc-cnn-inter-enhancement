import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    """Base configuration for the decoder."""

    bitstream_input: List[str] = field(default_factory=list)
    output_path: str = "./output/decoded"
    max_workers: Optional[int] = os.cpu_count()


BASE_CONFIG = Config()


@dataclass
class DecodingTaskParams:
    """Represents a single decoding job."""

    bitstream_input: str
    output_yuv: str
    trace_file: str
