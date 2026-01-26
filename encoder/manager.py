import re
from pathlib import Path
from typing import List
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from encoder.config import Config, EncodingTaskParams
from encoder.encoders import Encoder
from tqdm import tqdm

@dataclass
class Metadata:
    width: int
    height: int
    fps: int


@dataclass
class EncoderManager:
    cfg: Config
    encoder: Encoder

    def __post_init__(self):
        self.output_path = Path(self.cfg.output_dir).resolve()
        self.output_path.mkdir(parents=True, exist_ok=True)

    def _parse_info(self, info_path: Path) -> Metadata:
        content = info_path.read_text()
        width = int(re.search(r"width[:=\s]+(\d+)", content, re.I).group(1))
        height = int(re.search(r"height[:=\s]+(\d+)", content, re.I).group(1))
        fps_match = re.search(r"rate[:=\s]+([\d./]+)", content, re.I)

        fps_str = fps_match.group(1) if fps_match else "30"
        fps = round(eval(fps_str)) if "/" in fps_str else round(float(fps_str))

        return Metadata(width=width, height=height, fps=fps)

    def _generate_tasks(self) -> List[EncodingTaskParams]:
        tasks = []
        data_dir = Path(self.cfg.data_dir)

        for yuv_file in data_dir.glob("*.yuv"):
            info_candidates = list(data_dir.glob(f"{yuv_file.stem}*.info"))
            if not info_candidates:
                continue

            metadata = self._parse_info(info_candidates[0])

            for qp in self.cfg.qp:
                stem = yuv_file.stem

                task = EncodingTaskParams(
                    input_file=str(yuv_file),
                    width=metadata.width,
                    height=metadata.height,
                    fps=metadata.fps,
                    frames=self.cfg.frames_to_encode,
                    qp=qp,
                    bitstream_out=str(self.output_path / f"{stem}_QP{qp}.vvc"),
                    recon_out=str(self.output_path / f"{stem}_QP{qp}_rec.yuv"),
                    preset=self.cfg.preset,
                    alf=self.cfg.alf,
                    sao=self.cfg.sao,
                )
                tasks.append(task)
        return tasks

    def run(self):
        tasks = self._generate_tasks()
        print(
            f"Starting dataset generation: {len(tasks)} tasks using {self.cfg.max_workers} workers."
        )

        with ProcessPoolExecutor(max_workers=self.cfg.max_workers) as executor:
            results = list(executor.map(self.encoder.encode, tasks))
            future_to_task = {executor.submit(self.encoder.encode, t): t for t in tasks}

            # Add a progress bar logging
            with tqdm(total=len(tasks), desc="Encoding Videos", unit="task") as pbar:
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        res = future.result()
                        results.append(res)
                        pbar.set_postfix(
                            {"file": Path(task.input_file).stem, "qp": task.qp}
                        )
                    except Exception as e:
                        print(f"\n[ERROR] Task failed for {task.input_file}: {e}")

                    pbar.update(1)

        return results
