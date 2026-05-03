import os
from torch.utils.data import ConcatDataset, DataLoader, Dataset
import pytorch_lightning as pl
from pathlib import Path
import random
from typing import Optional, List
from enhancer.config import DataloaderConfig, DatasetConfig
from enhancer.vtm_dataset import VTMDataset


class VVCDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_config: DatasetConfig,
        dataloader_config: DataloaderConfig,
        test_full_frames: bool = False,
    ):
        super().__init__()
        self.config = dataloader_config
        self.dataset_config = dataset_config
        self.test_full_frames = test_full_frames

    def setup(self, stage: Optional[str] = None):
        all_videos = self._build_dataset_list(self.dataset_config.train_dir)

        random.seed(42)
        random.shuffle(all_videos)

        n = len(all_videos)
        train_end = int(n * 0.8)
        validate_end = int(n * 0.9)

        if stage == "fit" or stage is None:
            self.train_dataset = ConcatDataset(all_videos[:train_end])
            self.validate_dataset = ConcatDataset(all_videos[train_end:validate_end])
            print(f"✅ Training set: {len(self.train_dataset)} patches")

        if stage in ("test", "predict"):
            self.test_dataset = ConcatDataset(all_videos[validate_end:])

    def _build_dataset_list(self, directory: str) -> List[VTMDataset]:
        datasets = []
        base_path = Path(directory)
        orig_dir = Path("data")

        for dec_file in base_path.glob("*_rec.yuv"):
            base_stem = dec_file.name.replace("_vtm_rec.yuv", "").replace(
                "_rec.yuv", ""
            )
            video_name = base_stem.split("_QP")[0]

            trace_path = base_path / f"{base_stem}.csv"
            original_file = orig_dir / f"{video_name}.yuv"

            if trace_path.exists() and original_file.exists():
                try:
                    ds = VTMDataset(
                        decoded_yuv_filepath=str(dec_file),
                        original_yuv_filepath=str(original_file),
                        vtm_trace_path=str(trace_path),
                        patch_size=128,
                    )
                    if ds.width > 0 and ds.height > 0:
                        datasets.append(ds)
                except Exception as e:
                    print(f"❌ Error initializing {base_stem}: {e}")

        if not datasets:
            print("🚨 WARNING: No datasets were loaded.")
        else:
            print(f"✅ Successfully loaded {len(datasets)} videos.")

        return datasets

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Train dataset not initialized. Did you call setup()?")

        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.validate_dataset is None:
            raise RuntimeError("Val dataset not initialized. Did you call setup()?")

        return DataLoader(
            self.validate_dataset,
            batch_size=self.config.val_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.config.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not initialized. Did you call setup()?")

        t_batch = getattr(self.config, "test_batch_size", 1)
        batch_size = 1 if self.test_full_frames else t_batch

        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.config.num_workers,
        )
