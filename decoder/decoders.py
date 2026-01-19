from dataclasses import dataclass, field
import subprocess
import abc

from decoder.config import DecodingTaskParams


@dataclass
class Decoder(abc.ABC):
    """Abstract Base Class for Video Decoder."""

    executable: str

    @abc.abstractmethod
    def decode(self, task: DecodingTaskParams) -> str:
        """Decodes bitstream and extracts block-level statistics."""
        pass


@dataclass
class VTMDecoder(Decoder):
    executable: str = field(
        default="./bin/vtm/bin/umake/clang-15.0/x86_64/release/DecoderAnalyserApp",
    )

    def decode(self, task: DecodingTaskParams) -> str:
        """
        Decodes bitstream and extracts block-level statistics.
        """
        cmd = [
            self.executable,
            "-b",
            task.bitstream_input,
            "-o",
            task.output_yuv,
            f"--TraceFile={task.trace_file}",
            "--TraceRule=D_BLOCK_STATISTICS_ALL:poc>=0",
            "--OutputBitDepth=8",  # Ensure 8-bit output to match input
        ]

        # subprocess.run(
        #     cmd,
        #     stdout=subprocess.DEVNULL,
        #     stderr=subprocess.PIPE,
        #     text=True,
        #     check=True,
        # )
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print("\n--- VTM Error Output ---")
            print(e.stderr)  
            print("------------------------")
            raise e
        return task.trace_file
