# Martell Model Restore Instructions

## Branch: martell
## Date: 2026-05-03
## Python Version: 3.14.3

## Files Backed Up:
- `train_martell.py` - Training script
- `enhancer/` - Complete enhancer module (models, datasets, config, trainer)
- `precompute_features.py` - Precompute script for data generation
- `martell_epoch_190.pt` - Model weights (epoch 190, latest available)
- `pyproject.toml` - Project dependencies
- `uv.lock` - Exact dependency versions
- `snow_wide.py` - Model architecture
- `dataset_blackfyre.py` - Dataset loader

## Environment Setup:

### Option 1: Using uv (recommended)
```bash
uv venv --python 3.14
source .venv/bin/activate
uv pip install -e .
```

### Option 2: Using pip
```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Data Regeneration:
The precomputed data is in `data/precomputed_martell/` (too large to backup, ~55MB per sample).

To regenerate:
```bash
python scripts/precompute_features.py --input-dir /path/to/vvc/videos --output-dir data/precomputed_martell
```

## Model Architecture:
- **Model**: SnowWideEnhancer (from `enhancer/models/snow_wide.py`)
- **Config**: metadata_channels=9, base_channels=64
- **Dataset**: BlackfyreDataset (from `enhancer/dataset_blackfyre.py`)

## Loading Model for Testing:
```python
import torch
from enhancer.models.snow_wide import SnowWideEnhancer

class SnowWideConfig:
    metadata_channels = 9
    base_channels = 64

config = SnowWideConfig()
model = SnowWideEnhancer(config)
model.load_state_dict(torch.load('backup_martell/martell_epoch_190.pt'))
model.eval()
```

## Training Command:
```bash
python train_martell.py --batch-size 8 --epochs 500 --lr 1e-4
```

## Notes:
- No epoch 200 checkpoint exists yet (latest is 190)
- Checkpoint saves every 10 epochs
- Uses wandb for logging (offline mode)
- Patch size: 132
- Device: CUDA if available
