from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List

from decoder.config import Config, DecodingTaskParams
from decoder.decoders import Decoder


@dataclass
class DecoderManager:
    cfg: Config
    decoder: Decoder

    def _generate_tasks(self) -> List[DecodingTaskParams]:
        tasks: List[DecodingTaskParams] = []
        for b in self.cfg.bitstream_path:
            b_path = Path(b)
            tasks.append(
                DecodingTaskParams(
                    bitstream_path=str(b_path),
                    output_yuv=str(b_path.with_suffix(".vtm_rec.yuv")),
                    trace_file=str(b_path.with_suffix(".csv")),
                )
            )

        return tasks

    def run(self):
        """
        Takes a list of .vvc files and generates
        reconstructions + metadata traces.
        """
        tasks = self._generate_tasks()

        print(f"Starting VTM Metadata Extraction: {len(tasks)} tasks.")

        with ProcessPoolExecutor(max_workers=self.cfg.max_workers) as executor:
            results = list(executor.map(self.decoder.decode, tasks))

        for r in results:
            print(r)
