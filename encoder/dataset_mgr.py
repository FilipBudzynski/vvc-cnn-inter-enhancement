import yaml
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from encoders import Encoder, VVencFFapp
from dataclasses import dataclass


@dataclass
class DatasetManager:
    def __init__(self, config_path: str, encoder: Encoder):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.encoder = encoder
        self.output_path = Path(self.cfg["paths"]["output_dir"])
        self.output_path.mkdir(parents=True, exist_ok=True)

    def _parse_info(self, info_path):
        """Extracts metadata from .info files."""
        content = info_path.read_text()
        return {
            "width": int(re.search(r"Width\s+:\s+(\d+)", content).group(1)),
            "height": int(re.search(r"Height\s+:\s+(\d+)", content).group(1)),
            "fps": round(
                float(re.search(r"Frame rate\s+:\s+([\d.]+)", content).group(1))
            ),
        }

    def _generate_tasks(self):
        """Creates the list of encoding configurations (the 'Grid Search')."""
        tasks = []
        data_dir = Path(self.cfg["paths"]["data_dir"])

        for yuv_file in data_dir.glob("*.yuv"):
            info_file = list(data_dir.glob(f"{yuv_file.stem}*.info"))[0]

            if not info_file.exists():
                continue

            meta = self._parse_info(info_file)

            # Nested loops to create permutations based on YAML
            for qp in self.cfg["encoding_params"]["qp"]:
                stem = yuv_file.stem
                suffix = f"QP{qp}"

                tasks.append(
                    {
                        "input_file": str(yuv_file),
                        "width": meta["width"],
                        "height": meta["height"],
                        "fps": meta["fps"],
                        "frames": self.cfg["encoding_params"]["frames_to_encode"],
                        "qp": qp,
                        "bitstream_out": str(self.output_path / f"{stem}_{suffix}.vvc"),
                        "recon_out": str(self.output_path / f"{stem}_{suffix}_rec.yuv"),
                        "preset": self.cfg["encoding_params"].get("preset", "fast"),
                    }
                )
        return tasks

    def run(self):
        tasks = self._generate_tasks()
        print(f"Starting dataset generation: {len(tasks)} tasks.")

        with ProcessPoolExecutor(
            max_workers=self.cfg["execution"]["max_workers"]
        ) as executor:
            results = list(executor.map(self.encoder.encode, tasks))

        for r in results:
            print(r)


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    vvenc = VVencFFapp(executable_path=config["paths"]["encoder_path"])
    manager = DatasetManager("config.yaml", vvenc)
    manager.run()
