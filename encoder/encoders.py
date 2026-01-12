from dataclasses import dataclass, field
import subprocess
import abc
from pathlib import Path
from typing import TypeAlias

from encoder.config import EncodingTaskParams


class Encoder(abc.ABC):
    """Abstract Base Class for Video Encoder."""

    @abc.abstractmethod
    def encode(self, task: EncodingTaskParams) -> str:
        """Execute the encoding process for a specific task."""
        pass


@dataclass
class VVencEncoder(Encoder):
    executable: str = field(default="./bin/vvenc/bin/release-static/vvencFFapp")

    def encode(self, task: EncodingTaskParams) -> str:
        cmd = [
            self.executable,
            "-i", task.input_file,
            "-s", f"{task.width}x{task.height}",
            "-r", str(task.fps),
            "-f", str(task.frames),
            "-q", str(task.qp),
            "-b", task.bitstream_out,
            "-o", task.recon_out,
            "--preset", task.preset,
            "--alf", str(task.alf),
            "--sao", str(task.sao),
        ]

        log_path = Path(task.bitstream_out).with_suffix(".log")
        with open(log_path, "w") as log_file:
            subprocess.run(
                cmd, stdout=log_file, stderr=subprocess.PIPE, text=True, check=True
            )
        return task.bitstream_out
