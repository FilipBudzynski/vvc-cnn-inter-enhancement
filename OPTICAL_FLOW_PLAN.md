# Optical Flow Warping Implementation Plan

## Overview

This document details the implementation of **Optical Flow Warping** for inter-frame video enhancement, following the approach used in SOTA video enhancement networks.

### What is Optical Flow Warping?

Optical Flow Warping is a technique to align neighboring frames before processing them together:

1. **Estimate Motion**: Compute optical flow between frames (how pixels move)
2. **Warp Frames**: Shift pixels in neighbor frames based on flow to align with current frame
3. **Process Aligned Frames**: Use aligned frames for enhancement

### Why This Approach?

- **Simpler than DCNv2**: No custom CUDA compilation needed
- **Explicit Motion Handling**: Clear two-stage pipeline (flow → enhance)
- **Well-studied**: Many pre-trained flow models available (RAFT, FlowNet, etc.)
- **Proven Results**: Used in many SOTA video enhancement papers

---

## Architecture

### Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT                                    │
│  Frame F-1          Frame F0          Frame F+1                │
│  [B,3,H,W]         [B,3,H,W]         [B,3,H,W]                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   OPTICAL FLOW ESTIMATION                        │
│  ┌──────────────────┐     ┌──────────────────┐               │
│  │  Flow(F0 → F-1)  │     │  Flow(F0 → F+1)  │               │
│  │  [B, 2, H, W]   │     │  [B, 2, H, W]   │               │
│  └──────────────────┘     └──────────────────┘               │
│         ↓                          ↓                           │
│  Warp(F-1, flow)            Warp(F+1, flow)                  │
│  [B, 3, H, W]              [B, 3, H, W]                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   TEMPORAL FUSION                               │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  [F0, warped_F-1, warped_F+1, MV, metadata]         │    │
│  │  [B, 3 + 3 + 3 + 4 + 4 = 17 channels]                │    │
│  └────────────────────────────────────────────────────────┘    │
│                              │                                  │
│                              ▼                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              HightowerNet / ResNet                     │    │
│  │              (conv blocks → output)                    │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         OUTPUT                                  │
│  Enhanced F0: [B, 3, H, W]                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Details

### 1. Dependencies

**Required packages:**
```python
# Already available
torch
torchvision

# Need to install
pip install torchvision  # For RAFT model
```

**RAFT Model (from torchvision):**
- Pre-trained model available in torchvision
- Two variants: `raft_small()` and `raft_large()`
- We'll use `raft_small()` for speed

### 2. New Files to Create

| File | Purpose |
|------|---------|
| `enhancer/models/optical_flow.py` | Optical flow estimation + warping utilities |
| `enhancer/dataset_optical_flow.py` | Dataset with pre-computed flow |
| `train_optical_flow.py` | Training script |

### 3. Modified Files

| File | Changes |
|------|---------|
| `enhancer/models/hightower.py` | Add warping support |

---

## Step-by-Step Implementation

### Step 1: Create Optical Flow Module

**File: `enhancer/models/optical_flow.py`**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_small, RAFT


class FlowEstimator(nn.Module):
    """Pre-trained RAFT model for optical flow estimation"""
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.model = raft_small(pretrained=pretrained)
        self.model.eval()
        
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Estimate optical flow from img1 to img2
        
        Args:
            img1: [B, 3, H, W] - source frame
            img2: [B, 3, H, W] - target frame
            
        Returns:
            flow: [B, 2, H, W] - flow field (dx, dy) in pixel coordinates
        """
        # Normalize to [-1, 1]
        img1 = img1 * 2 - 1
        img2 = img2 * 2 - 1
        
        with torch.no_grad():
            flow_pred = self.model(img1, img2)
            # RAFT returns list of flows, get final one
            flow = flow_pred[-1]  # [B, 2, H', W']
        
        # Resize flow to match input resolution if needed
        if flow.shape[2:] != img1.shape[2:]:
            flow = F.interpolate(
                flow, 
                size=img1.shape[2:], 
                mode='bilinear', 
                align_corners=True
            )
            # Scale flow to match the new resolution
            flow = flow * torch.tensor(
                [img1.shape[3] / flow.shape[3], img1.shape[2] / flow.shape[2]],
                device=flow.device
            ).view(1, 2, 1, 1)
        
        return flow


def warp_frame(frame: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Warp a frame using optical flow
    
    Args:
        frame: [B, C, H, W] - frame to warp
        flow: [B, 2, H, W] - flow field (dx, dy) at each pixel
        
    Returns:
        warped: [B, C, H, W] - warped frame
    """
    B, C, H, W = frame.shape
    
    # Create grid for sampling
    # y_coords: [0, 1, 2, ..., H-1]
    # x_coords: [0, 1, 2, ..., W-1]
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=flow.device),
        torch.arange(W, device=flow.device),
        indexing='ij'
    )
    
    # Add flow to coordinates
    # flow[:, 0, :, :] = horizontal displacement (x)
    # flow[:, 1, :, :] = vertical displacement (y)
    grid_y = y_coords.float() + flow[:, 1, :, :]  # [B, H, W]
    grid_x = x_coords.float() + flow[:, 0, :, :]  # [B, H, W]
    
    # Normalize to [-1, 1] for grid_sample
    grid_y = 2.0 * grid_y / (H - 1) - 1.0
    grid_x = 2.0 * grid_x / (W - 1) - 1.0
    
    # Stack to [B, H, W, 2] (y, x order for grid_sample)
    grid = torch.stack([grid_x, grid_y], dim=3)  # [B, H, W, 2]
    
    # Sample from frame
    warped = F.grid_sample(
        frame,
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    
    return warped
```

### Step 2: Create Optical Flow Dataset

**File: `enhancer/dataset_optical_flow.py`**

```python
"""
Dataset that pre-computes optical flow for faster training
"""

import random
import torch
from torch.utils.data import Dataset
from pathlib import Path


class OpticalFlowDataset(Dataset):
    """
    Dataset with pre-computed optical flow
    
    For efficiency, we can either:
    1. Pre-compute flow during dataset creation (slower init, faster training)
    2. Compute flow on-the-fly (faster init, slower training)
    
    We'll use option 2 for simplicity, with caching.
    """
    
    def __init__(
        self,
        data_dir: str = "data/precomputed",
        patch_size: int = 132,
        split: str = "train",
        train_ratio: float = 0.8,
        use_flow_warping: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.split = split
        self.use_flow_warping = use_flow_warping
        
        # Group frames by video
        self.video_frames = {}
        for video_dir in self.data_dir.iterdir():
            if video_dir.is_dir():
                frames = sorted(video_dir.glob("poc_*.pt"), 
                             key=lambda x: int(x.stem.split("_")[1]))
                if len(frames) >= 3:
                    self.video_frames[video_dir.name] = frames
        
        # Create valid indices (frame index 1 to n-2 for F-1, F0, F+1)
        self.samples = []
        for video_name, frames in self.video_frames.items():
            for i in range(1, len(frames) - 1):
                self.samples.append((video_name, frames[i-1], frames[i], frames[i+1]))
        
        # Shuffle and split
        random.seed(42)
        random.shuffle(self.samples)
        
        n = len(self.samples)
        if split == "train":
            self.samples = self.samples[:int(n * train_ratio)]
        elif split == "val":
            self.samples = self.samples[int(n * train_ratio):int(n * (train_ratio + 0.1))]
        else:
            self.samples = self.samples[int(n * (train_ratio + 0.1)):]
        
        print(f"Loaded {len(self.samples)} frame triplets for {split}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_name, prev_pt, curr_pt, next_pt = self.samples[idx]
        
        # Load frames
        prev_data = torch.load(prev_pt)
        curr_data = torch.load(curr_pt)
        next_data = torch.load(next_pt)
        
        # Extract YUV frames
        prev_frame = prev_data["decoded"]
        curr_frame = curr_data["decoded"]
        next_frame = next_data["decoded"]
        original = curr_data["original"]
        
        # Features (motion vectors + metadata)
        features = curr_data["features"]
        motion_vectors = features[:4]
        metadata = features[4:]
        
        # Random crop
        _, h, w = curr_frame.shape
        th, tw = self.patch_size, self.patch_size
        
        if h > th:
            top = random.choice(range(0, h - th + 1, 8))
        else:
            top = 0
        if w > tw:
            left = random.choice(range(0, w - tw + 1, 8))
        else:
            left = 0
        
        prev_frame = prev_frame[:, top:top+th, left:left+tw]
        curr_frame = curr_frame[:, top:top+th, left:left+tw]
        next_frame = next_frame[:, top:top+th, left:left+tw]
        original = original[:, top:top+th, left:left+tw]
        motion_vectors = motion_vectors[:, top:top+th, left:left+tw]
        metadata = metadata[:, top:top+th, left:left+tw]
        
        # For optical flow: we're not pre-computing flow here
        # Flow will be computed on-the-fly in the training loop
        return (prev_frame, curr_frame, next_frame), original, motion_vectors, metadata, {
            "video": video_name,
            "poc": curr_data["poc"]
        }
```

### Step 3: Update Hightower Model

**File: `enhancer/models/hightower.py`** - Add warping support

Add these imports and modify forward:

```python
from .optical_flow import FlowEstimator, warp_frame


class HightowerNetWithWarping(nn.Module):
    """
    Hightower with Optical Flow Warping
    
    Key difference from HightowerNet:
    - Warps neighboring frames using optical flow before concatenation
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Base Hightower
        self.base_model = HightowerNet(...)
        
        # Optical flow estimator (pre-trained RAFT)
        self.flow_estimator = FlowEstimator(pretrained=True)
        self.flow_estimator.eval()
        
        # Freeze flow estimator (don't train it)
        for param in self.flow_estimator.parameters():
            param.requires_grad = False
    
    def forward(self, frames: list, motion_vectors: Tensor, metadata: Tensor) -> Tensor:
        """
        frames: [F-1, F0, F+1]
        """
        prev_frame, curr_frame, next_frame = frames
        
        # Estimate optical flow
        # Flow from F0 to F-1 (how to get from F-1 to F0)
        flow_prev = self.flow_estimator(prev_frame, curr_frame)  # [B, 2, H, W]
        
        # Flow from F0 to F+1
        flow_next = self.flow_estimator(curr_frame, next_frame)
        
        # Warp neighboring frames to align with current frame
        warped_prev = warp_frame(prev_frame, flow_prev)
        warped_next = warp_frame(next_frame, flow_next)
        
        # Use warped frames instead of original
        aligned_frames = [warped_prev, curr_frame, warped_next]
        
        # Process with base model
        return self.base_model(aligned_frames, motion_vectors, metadata)
```

### Step 4: Create Training Script

**File: `train_optical_flow.py`**

```python
#!/usr/bin/env python3
"""
Optical Flow Warping Training Script
Branch: optical-flow

This version uses optical flow warping to align neighboring frames
before enhancement.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

import wandb
from enhancer.models.hightower import HightowerNetWithWarping
from enhancer.models.optical_flow import FlowEstimator, warp_frame
from enhancer.ssim import SSIM, MS_SSIM
from enhancer.config import Config
from enhancer.dataset_optical_flow import OpticalFlowDataset

# Same loss function as Hightower
# Same training loop structure
# Key difference: compute flow on-the-fly

def compute_loss(enhanced, original, ssim_mod, msssim_mod):
    """Same loss as Hightower"""
    # ... (same as before)


def main():
    # ... (similar setup)
    
    # Model with optical flow
    model = HightowerNetWithWarping(config).to(DEVICE)
    
    # Flow estimator (frozen)
    flow_estimator = FlowEstimator(pretrained=True).to(DEVICE)
    flow_estimator.eval()
    
    for epoch in range(args.epochs):
        # ... training loop ...
        
        # In forward pass:
        # 1. Estimate flow
        flow_prev = flow_estimator(prev_frames, curr_frames)
        flow_next = flow_estimator(curr_frames, next_frames)
        
        # 2. Warp frames
        warped_prev = warp_frame(prev_frames, flow_prev)
        warped_next = warp_frame(next_frames, flow_next)
        
        # 3. Use warped frames for enhancement
        enhanced = model([warped_prev, curr_frames, warped_next], 
                        motion_vectors, metadata)


if __name__ == "__main__":
    main()
```

---

## Alternative: Pre-computed Flow

For faster training, pre-compute flow during dataset creation:

```python
# In dataset __getitem__:
# After loading frames, compute flow once and cache
if self.use_flow_warping and self.split == "train":
    with torch.no_grad():
        flow_prev = flow_estimator(prev_frame.unsqueeze(0), curr_frame.unsqueeze(0))
        flow_next = flow_estimator(curr_frame.unsqueeze(0), next_frame.unsqueeze(0))
    return frames, original, flow_prev, flow_next, ...
```

This trades init time for training speed.

---

## Performance Considerations

### Speed vs Accuracy Tradeoffs

| Option | Speed | Accuracy | Notes |
|--------|-------|----------|-------|
| RAFT Small | Fast | Good | Recommended |
| RAFT Large | Slow | Best | For final model |
| Pre-computed Flow | Fastest | Same | Best for training |
| On-the-fly Flow | Slower | Same | Simpler to implement |

### Memory Considerations

- RAFT model: ~6M parameters
- Flow estimation: ~1GB GPU memory
- Warping: Negligible overhead

### Training Time Estimate

- With on-the-fly flow: ~2x training time
- With pre-computed flow: ~1.3x training time

---

## Expected Results

Based on SOTA literature:

| Method | PSNR Gain |
|--------|-----------|
| Simple Concat (current) | +0.27 dB |
| Optical Flow Warping | +0.4 - 0.6 dB |
| DCNv2 (STDF) | +0.78 dB |

Optical Flow Warping should provide significant improvement over simple concatenation.

---

## Comparison with Current Approach

| Aspect | Current (Concat) | Optical Flow Warping |
|--------|-----------------|---------------------|
| Frame Alignment | None (implicit) | Explicit (flow-based) |
| Motion Handling | Network learns | Pre-compensated |
| Complexity | Low | Medium |
| Extra Params | None | ~6M (RAFT) |
| Expected Gain | +0.27 dB | +0.4-0.6 dB |

---

## References

1. **RAFT**: "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow" (ECCV 2020)
2. **STDF**: "Spatio-Temporal Deformable Convolution for Compressed Video Quality Enhancement" (AAAI 2020)
3. **FDAN**: "Flow-guided Deformable Alignment Network for Video Super-Resolution" (CVPR 2021)
4. **Torchvision RAFT**: `torchvision.models.optical_flow.raft_small`

---

## Implementation Checklist

- [ ] Install requirements (torchvision)
- [ ] Create `enhancer/models/optical_flow.py`
- [ ] Create `enhancer/dataset_optical_flow.py`
- [ ] Update `enhancer/models/hightower.py` with warping support
- [ ] Create `train_optical_flow.py`
- [ ] Test on small dataset
- [ ] Train full model
- [ ] Compare results

---

## Open Questions

1. **Flow estimation frequency**: Should we compute flow every batch or every N batches?
2. **Flow model**: Use RAFT small or large?
3. **Pre-compute vs on-the-fly**: Which approach for your thesis?
4. **Mixed precision**: Use FP16 for flow estimation to speed up?

Please review and let me know which options you prefer!
