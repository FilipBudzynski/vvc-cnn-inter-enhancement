import os

# import torch
from torch.utils.data import DataLoader

# from torchvision import transforms
import pytorch_lightning as pl
from config import DataloaderConfig, DatasetConfig

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
        Swapping VVCDataset for VTMDataset.
        We assume your dataset_config (from config.yaml) now contains:
        - dec_yuv, orig_yuv, trace, width, height
        """
        if stage == "fit":
            self.train_dataset = VTMDataset(
                decoded_yuv_filepath=self.dataset_config.train.decoded_yuv_filepath,
                original_yuv_filepath=self.dataset_config.train.original_yuv_file_path,
                vtm_trace_path=self.dataset_config.train.vtm_trace_filepath,
                width=self.dataset_config.train.width,
                height=self.dataset_config.train.height,
                chunk_transform=self.chunk_transform(),
            )

            # Virtual epoch calculation
            steps_per_real_epoch = len(self.train_dataset) / self.config.n_step
            print(f"Training: {steps_per_real_epoch:.2f} virtual steps per real epoch")

            self.validate_dataset = VTMDataset(
                decoded_yuv_filepath=self.dataset_config.val.decoded_yuv_filepath,
                original_yuv_filepath=self.dataset_config.val.original_yuv_file_path,
                vtm_trace_path=self.dataset_config.val.vtm_trace_filepath,
                width=self.dataset_config.val.width,
                height=self.dataset_config.val.height,
                chunk_transform=self.chunk_transform(),
            )

        if stage in ("test", "predict"):
            # For testing, we use the test split from config
            self.test_dataset = VTMDataset(
                decoded_yuv_filepath=self.dataset_config.test.decoded_yuv_filepath,
                original_yuv_filepath=self.dataset_config.test.original_yuv_file_path,
                vtm_trace_path=self.dataset_config.test.vtm_trace_filepath,
                width=self.dataset_config.test.width,
                height=self.dataset_config.test.height,
                chunk_transform=self.chunk_transform(),
            )

    def train_dataloader(self):
        assert (
            self.train_dataset is not None
        ), "train_dataset not initialized. Did you call setup()?"
        data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True,
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
