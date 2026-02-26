import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from .config import DataloaderConfig, DatasetConfig

from .vtm_dataset import VTMDataset as VVCDataset, VTMFrameDataset as FrameDataset


class LoaderWrapper:
    """LoaderWrapper."""

    def __init__(self, dataloader: DataLoader, n_step: int):
        """__init__.

        :param dataloader:
        :type dataloader: DataLoader
        :param n_step:
        :type n_step: int
        """
        self.n_step = n_step
        self.idx = 0
        self.dataloader = dataloader
        self.iter_loader = iter(dataloader)

    def __iter__(self) -> "LoaderWrapper":
        """__iter__.

        :rtype: "LoaderWrapper"
        """
        return self

    def __len__(self) -> int:
        """__len__.

        :rtype: int
        """
        return self.n_step

    def __next__(self):
        """__next__."""
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
    """VVCDataModule."""

    def __init__(
        self,
        dataset_config: DatasetConfig,
        dataloader_config: DataloaderConfig,
        test_full_frames: bool = False,
    ):
        """__init__.

        :param chunk_folder:
        :type chunk_folder: str
        :param orig_chunk_folder:
        :type orig_chunk_folder: str
        :param chunk_height:
        :type chunk_height: int
        :param chunk_width:
        :type chunk_width: int
        :param batch_size:
        :type batch_size: int
        """
        super().__init__()
        self.config = dataloader_config
        self.dataset_config = dataset_config

        self.dataset_val = None
        self.dataset_test = None
        self.dataset_train = None
        self.test_full_frames = test_full_frames

    def setup(self, stage=None):
        """setup.

        :param stage:
        """
        if stage == "fit":
            self.dataset_train = VVCDataset(
                settings=self.dataset_config.train,
                chunk_transform=self.chunk_transform(),
                metadata_transform=self.metadata_transform(),
            )

            epochs_for_real_one = len(self.dataset_train) / self.config.n_step
            print(f"it takes {epochs_for_real_one} of training to reach one real epoch")

            self.dataset_val = VVCDataset(
                settings=self.dataset_config.val,
                chunk_transform=self.chunk_transform(),
                metadata_transform=self.metadata_transform(),
            )

            epochs_for_real_one = len(self.dataset_val) / self.config.val_n_step
            print(
                f"it takes {epochs_for_real_one} of validation to reach one real epoch"
            )

        if stage in ("test", "predict"):
            if self.test_full_frames:
                self.dataset_test = FrameDataset(
                    settings=self.dataset_config.test,
                    chunk_transform=self.chunk_transform(),
                    metadata_transform=self.metadata_transform(),
                )
            else:
                self.dataset_test = VVCDataset(
                    settings=self.dataset_config.test,
                    chunk_transform=self.chunk_transform(),
                    metadata_transform=self.metadata_transform(),
                )

    def train_dataloader(self):
        """train_dataloader."""
        data_loader = DataLoader(
            self.dataset_train,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=os.cpu_count(),
        )
        return LoaderWrapper(
            data_loader,
            self.config.n_step,
        )

    def test_dataloader(self, shuffle=False):
        """test_dataloader."""
        data_loader = DataLoader(
            self.dataset_test,
            batch_size=self.config.test_batch_size if not self.test_full_frames else 1,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=os.cpu_count(),
        )
        return data_loader

    def predict_dataloader(self):
        return self.test_dataloader(True)

    def val_dataloader(self):
        """val_dataloader."""
        data_loader = DataLoader(
            self.dataset_val,
            batch_size=self.config.val_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=os.cpu_count(),
        )
        return LoaderWrapper(
            data_loader,
            self.config.val_n_step,
        )

    def chunk_transform(self):
        """chunk_transform."""
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        return transform

    def metadata_transform(self):
        """metadata_transform."""

        def transform(metadata):
            return torch.as_tensor(metadata).float().view(len(metadata), 1, 1)

        return transform