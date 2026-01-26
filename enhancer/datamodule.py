import os
import torch
from torch.utils.data import ConcatDataset, DataLoader
import pytorch_lightning as pl
from pathlib import Path
import random
from enhancer.config import DataloaderConfig, DatasetConfig
from enhancer.vtm_dataset import VTMDataset


class LoaderWrapper:
    def __init__(self, dataloader: DataLoader, n_step: int):
        self.n_step = n_step
        self.idx = 0
        self.dataloader = dataloader
        self.iter_loader = iter(dataloader)

    def __iter__(self) -> "LoaderWrapper":
        return self

    def __len__(self) -> int:
        return self.n_step

    def __next__(self):
        if self.idx == self.n_step:
            self.idx = 0
            raise StopIteration
        else:
            self.idx += 1

        try:
            return next(self.iter_loader)
        except StopIteration:
            self.iter_loader = iter(self.dataloader)
            return next(self.iter_loader)


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

        self.validate_dataset = None
        self.test_dataset = None
        self.train_dataset = None
        self.test_full_frames = test_full_frames

    def setup(self, stage=None):
        """
        We assume your dataset_config (from config.yaml) contains:
        - dec_yuv, orig_yuv, trace, width, height
        """
        all_videos = self._build_dataset_list(self.dataset_config.train_dir)
        random.seed(42)
        random.shuffle(all_videos)
        n = len(all_videos)
        train_end = int(n * 0.8)
        validate_end = int(n * 0.9)

        train_videos = all_videos[:train_end]
        validate_videos = all_videos[train_end:validate_end]
        test_videos = all_videos[validate_end:]

        if stage == "fit" or stage is None:
            self.train_dataset = ConcatDataset(train_videos)
            self.validate_dataset = ConcatDataset(validate_videos)

            steps_per_real_epoch = len(self.train_dataset) / self.config.n_step
            print(f"Training: {steps_per_real_epoch:.2f} virtual steps per real epoch")

        if stage in ("test", "predict"):
            self.test_dataset = ConcatDataset(test_videos)

    def _build_dataset_list(self, directory: str):
        """Helper to pair files and return a list of VTMDatasets"""
        datasets = []
        base_path = Path(directory)

        for dec_file in base_path.glob("*_rec.yuv"):
            stem = dec_file.name.split("_QP")[0]
            trace_file = dec_file.with_suffix(".csv").name.replace("_rec", "")
            trace_path = base_path / trace_file

            original_file = f"{stem}.yuv"
            datasets.append(
                VTMDataset(
                    decoded_yuv_filepath=str(dec_file),
                    original_yuv_filepath=os.path.join(
                        self.dataset_config.original_dir, original_file
                    ),
                    vtm_trace_path=str(trace_path),
                    patch_size=128,
                )
            )
        return datasets

    def train_dataloader(self):
        assert (
            self.train_dataset is not None
        ), "train_dataset not initialized. Did you call setup()?"
        data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
            num_workers=os.cpu_count() or 4,
        )
        return LoaderWrapper(data_loader, self.config.n_step)

    def val_dataloader(self):
        assert (
            self.validate_dataset is not None
        ), "validate_dataset not initialized. Did you call setup()?"
        data_loader = DataLoader(
            self.validate_dataset,
            batch_size=self.config.val_batch_size,
            shuffle=False,  # Usually better to keep val stable
            pin_memory=True,
            num_workers=os.cpu_count() or 4,
        )
        return LoaderWrapper(data_loader, self.config.val_n_step)

    def test_dataloader(self, shuffle=False):
        assert (
            self.test_dataset is not None
        ), "validate_dataset not initialized. Did you call setup()?"
        return DataLoader(
            self.test_dataset,
            batch_size=1 if self.test_full_frames else self.config.test_batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=os.cpu_count() or 4,
        )

    def chunk_transform(self):
        # Our VTMDataset already converts to Tensor and normalizes.
        # If you need extra Augmentation (like RandomCrops), add them here.
        return None
