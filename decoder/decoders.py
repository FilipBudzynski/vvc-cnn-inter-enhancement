import subprocess


class VTMDecoder:
    def __init__(self, executable_path: str):
        self.executable = executable_path

    def decode(self, bitstream_path: str, output_yuv: str, trace_file: str) -> str:
        """
        Decodes bitstream and extracts block-level statistics.
        """
        cmd = [
            self.executable,
            "-b",
            bitstream_path,
            "-o",
            output_yuv,
            f"--TraceFile={trace_file}",
            "--TraceRule=D_BLOCK_STATISTICS:all",
        ]

        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return trace_file
