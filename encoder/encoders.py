import subprocess
import abc
from pathlib import Path


class Encoder(abc.ABC):
    """Abstract Base Class for Video Encoder."""

    @abc.abstractmethod
    def encode(self, task: dict) -> str:
        """Execute the encoding process for a specific task."""
        pass


class VVencFFapp(Encoder):
    def __init__(self, executable_path: str):
        self.executable = executable_path

    def encode(self, task: dict) -> str:
        # vvencFFapp standard CLI
        cmd = [
            self.executable,
            "-i",
            task["input_file"],
            "-s",
            f"{task['width']}x{task['height']}",
            "-r",
            str(task["fps"]),
            "-f",
            str(task["frames"]),
            "-q",
            str(task["qp"]),
            "-b",
            task["bitstream_out"],
            "-o",
            task["recon_out"],
            "--preset",
            task.get("preset", "fast"),
            "--alf",
            str(task.get("alf", 1)),
            "--sao",
            str(task.get("sao", 1)),
        ]

        log_path = Path(task["bitstream_out"]).with_suffix(".log")
        with open(log_path, "w") as log_file:
            subprocess.run(
                cmd, stdout=log_file, stderr=subprocess.PIPE, text=True, check=True
            )
        return task["bitstream_out"]
