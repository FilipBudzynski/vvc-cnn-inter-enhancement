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
        self.output_path = Path(self.cfg.output_path).resolve()
        self.output_path.mkdir(parents=True, exist_ok=True)

        for b in self.cfg.bitstream_input:
            b_path = Path(b)
            file_name = b_path.stem
            tasks.append(
                DecodingTaskParams(
                    bitstream_input=str(b_path),
                    output_yuv=str(self.output_path / f"{file_name}_vtm_rec.yuv"),
                    trace_file=str(self.output_path / f"{file_name}.csv"),
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
