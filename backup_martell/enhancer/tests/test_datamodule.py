import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch
from enhancer.datamodule import VVCDataModule

class MockVTMDataset:
    def __init__(self, name): self.name = name
    def __len__(self): return 10
    def __getitem__(self, idx): 
        return torch.randn(10, 24, 24), torch.randn(3, 24, 24), {}


@pytest.fixture
def mock_configs():
    """Creates mock config objects to avoid needing a real yaml file."""
    dataset_cfg = MagicMock()
    dataset_cfg.train_dir = "data/decoded"
    dataset_cfg.original_dir = "data/originals"

    dataloader_cfg = MagicMock()
    dataloader_cfg.batch_size = 2
    dataloader_cfg.n_step = 10
    dataloader_cfg.val_batch_size = 2
    dataloader_cfg.val_n_step = 2

    return dataset_cfg, dataloader_cfg


def test_datamodule_setup(mock_configs):
    dataset_config, dataloader_config = mock_configs

    datamodule = VVCDataModule(
        dataset_config=dataset_config, dataloader_config=dataloader_config
    )

    fake_videos = [MockVTMDataset(f"video_{i}") for i in range(10)]

    with patch.object(VVCDataModule, '_build_dataset_list', return_value=fake_videos):
        datamodule.setup(stage="fit")

        assert len(datamodule.train_dataset.datasets) == 8
        assert len(datamodule.validate_dataset.datasets) == 1
        
        loader = datamodule.train_dataloader()
        x, y, info = next(iter(loader))
        
        assert x.shape == (2, 10, 24, 24)
        assert y.shape == (2, 3, 24, 24)
