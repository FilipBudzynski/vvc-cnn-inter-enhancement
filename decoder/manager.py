from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List

from decoder.config import Config, DecodingTaskParams
from decoder.decoders import Decoder
from tqdm import tqdm


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

        all_results = []
        with ProcessPoolExecutor(max_workers=self.cfg.max_workers) as executor:
            # results = list(executor.map(self.decoder.decode, tasks))
            future_to_task = {executor.submit(self.decoder.decode, t): t for t in tasks}


            with tqdm(total=len(tasks), desc="Decoding & Tracing", unit="file") as pbar:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        res = future.result()
                        all_results.append(res)
                        pbar.set_postfix({"file": Path(task.bitstream_input).name})
                    except Exception as e:
                        print(
                            f"\n[ERROR] Decoding failed for {task.bitstream_input}: {e}"
                        )

                    pbar.update(1)
        return all_results
