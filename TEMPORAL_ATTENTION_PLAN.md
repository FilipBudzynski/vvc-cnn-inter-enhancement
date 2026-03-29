# Temporal Attention with Enhanced Decoder Metadata - Implementation Plan

## Branch: blackfyre
## Named after House Blackfyre from Game of Thrones

---

## Overview

This plan implements **Temporal Attention** with enhanced decoder metadata utilization for VVC video enhancement. Instead of using Optical Flow (which doesn't leverage decoder information), we use:

1. **Temporal Attention** - Learn which frames/regions to focus on
2. **Enhanced Metadata** - Use more decoder information (QP, Depth, CTU info, etc.)
3. **Metadata-Guided Attention** - Use QP/Depth as attention weights

---

## Key Papers Referenced

### 1. "Prediction-Aware Quality Enhancement Framework for VVC" (IEEE 2024)
- **URL**: https://ieeexplore.ieee.org/document/9465693
- **Key Idea**: Use QP, motion vectors, prediction mode directly as input features
- **Our Implementation**: Extend with temporal attention

### 2. "Multi-Frame Quality Enhancement Model for VVC" (IEEE 2024)
- **URL**: https://ieeexplore.ieee.org/document/9255448
- **Key Idea**: Multiple frames + decoder metadata together
- **Our Implementation**: Use F-1, F0, F+1 with attention weights

### 3. "Attention-Based Dual-Scale CNN In-Loop Filter for VVC"
- **URL**: https://ieeexplore.ieee.org/ielx7/6287639/8600701/08852743.pdf
- **Key Idea**: Attention mechanism for different scales
- **Our Implementation**: Spatial attention + temporal attention

---

## Current Hightower vs New Blackfyre

| Aspect | Hightower (Current) | Blackfyre (New) |
|--------|---------------------|-----------------|
| **Frames** | F-1, F0, F+1 concat | F-1, F0, F+1 with attention |
| **Metadata** | 8 channels | 16+ channels (enhanced) |
| **Fusion** | Simple concat | Weighted by learned attention |
| **Attention** | None | Temporal + Spatial |

---

## Enhanced Metadata Utilization

### Currently Used (8 channels):
```
[MVL0_X, MVL0_Y, MVL1_X, MVL1_Y, QP, Depth, PredMode, Boundary]
```

### Additional Metadata to Add (8+ more):
```
QT_Depth      - QuadTree depth (partition complexity)
BT_Depth      - BinaryTree depth  
SkipFlag      - Skip mode (often indicates smooth areas)
MergeFlag     - Merge mode
InterDir      - Inter direction (L0/L1/Bi)
RefIdxL0      - Reference frame index L0
RefIdxL1      - Reference frame index L1
Cbf_Y         - Luma coded block flag
```

### Metadata as Attention Weights:
- QP map → Where to enhance more (high QP = more enhancement)
- Depth → Where complexity is high
- SkipFlag → Don't enhance skipped blocks (already good)

---

## Architecture: Temporal Attention Network

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT                                    │
│  Frames: [F-1, F0, F+1]          Metadata: [QP, Depth, MV...] │
│  [B, 3, 3, H, W]                 [B, 16, H, W]                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FRAME ENCODING                                  │
│  Each frame → Conv2D → Feature map [B, 64, H, W]               │
│                                                                 │
│  F-1_features = encoder(F-1)  [B, 64, H, W]                   │
│  F0_features  = encoder(F0)    [B, 64, H, W]                  │
│  F+1_features = encoder(F+1)  [B, 64, H, W]                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────�│
│                 TEMPORAL ATTENTION LAYER                          │
│                                                                 │
│  Query: F0_features                                             │
│  Keys:   [F-1_features, F0_features, F+1_features]              │
│  Values: [F-1_features, F0_features, F+1_features]              │
│                                                                 │
│  Attention = softmax(Q @ K^T / sqrt(d))                        │
│  Output = Attention @ V                                          │
│                                                                 │
│  Additionally, modulate by QP/Depth metadata!                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────�│
│                 METADATA ENHANCEMENT                              │
│                                                                 │
│  Use QP as attention mask:                                      │
│  - High QP regions → higher attention weight                    │
│  - Low QP regions → lower attention weight                      │
│                                                                 │
│  Use Depth as feature modulation                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         OUTPUT                                    │
│  Enhanced F0: [B, 3, H, W]                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Details

### Step 1: Update Parser for Enhanced Metadata

**File: `features_parser/parser.py`**

Add new handlers for additional metadata:
```python
# Additional metadata to parse
ADDITIONAL_PARAMS = [
    "QT_Depth",
    "BT_Depth", 
    "SkipFlag",
    "MergeFlag",
    "InterDir",
    "RefIdxL0", 
    "RefIdxL1",
    "Cbf_Y",
]
```

### Step 2: Update Feature Generator

**File: `features_generator/generator.py`**

Generate more feature channels:
```python
# Expanded feature order
FEATURE_ORDER = [
    # Motion vectors (4)
    "QP", "PredMode", "Depth", "Boundary",  # Current (4)
    "MVL0_X", "MVL0_Y", "MVL1_X", "MVL1_Y",  # Motion (4)
    # NEW: Enhanced metadata (8+)
    "QT_Depth", "BT_Depth",
    "SkipFlag", "MergeFlag", 
    "InterDir", "RefIdxL0", "RefIdxL1",
    "Cbf_Y",
]
```

### Step 3: Create Temporal Attention Model

**File: `enhancer/models/blackfyre.py`**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Learn which frames to attend to"""
    
    def __init__(self, channels: int = 64):
        super().__init__()
        self.query_conv = nn.Conv2d(channels, channels, 1)
        self.key_conv = nn.Conv2d(channels, channels, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable weight
        
    def forward(self, current, prev, next):
        # current, prev, next: [B, C, H, W]
        
        # Compute query, keys, values
        q = self.query_conv(current)  # [B, C, H, W]
        k_prev = self.key_conv(prev)
        k_next = self.key_conv(next)
        k_current = self.key_conv(current)
        
        # Keys from all frames
        keys = torch.stack([k_prev, k_current, k_next], dim=1)  # [B, 3, C, H, W]
        values = torch.stack([prev, current, next], dim=1)  # [B, 3, C, H, W]
        
        # Attention (simplified)
        B, C, H, W = q.shape
        q_flat = q.view(B, C, -1)  # [B, C, HW]
        k_flat = keys.view(B, 3, C, -1)  # [B, 3, C, HW]
        
        # Attention scores
        attn = torch.einsum('bcn,bcmn->bmn', q_flat, k_flat)  # [B, 3, HW]
        attn = F.softmax(attn, dim=1)
        
        # Apply attention to values
        v_flat = values.view(B, 3, C, -1)  # [B, 3, C, HW]
        out_flat = torch.einsum('bmn,bcmn->bcn', attn, v_flat)  # [B, C, HW]
        out = out_flat.view(B, C, H, W)
        
        # Residual connection
        return current + self.gamma * out


class MetadataAttention(nn.Module):
    """Use metadata (QP, Depth) to modulate features"""
    
    def __init__(self, metadata_channels: int, feature_channels: int):
        super().__init__()
        self.metadata_conv = nn.Conv2d(metadata_channels, feature_channels, 1)
        self.attention_conv = nn.Conv2d(feature_channels, 1, 1)
        
    def forward(self, features, metadata):
        # features: [B, C, H, W]
        # metadata: [B, M, H, W]
        
        # Transform metadata to same channel
        meta_feat = self.metadata_conv(metadata)  # [B, C, H, W]
        
        # Compute attention from features + metadata
        combined = features + meta_feat
        attn_weights = torch.sigmoid(self.attention_conv(combined))  # [B, 1, H, W]
        
        # Apply attention
        return features * attn_weights


class BlackfyreNet(nn.Module):
    """Temporal Attention + Enhanced Metadata Network"""
    
    def __init__(self, num_frames: int = 3, metadata_channels: int = 16):
        super().__init__()
        
        # Frame encoder
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.PReLU(),
        )
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(channels=64)
        
        # Metadata processing
        self.metadata_attention = MetadataAttention(metadata_channels, 64)
        
        # Main processing
        self.blocks = nn.Sequential(
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
        )
        
        # Output
        self.output_conv = nn.Conv2d(64, 3, 3, padding=1)
        
    def forward(self, frames, metadata):
        # frames: [F-1, F0, F+1] list of [B, 3, H, W]
        # metadata: [B, M, H, W]
        
        # Encode each frame
        encoded = [self.frame_encoder(f) for f in frames]
        
        # Temporal attention: focus on relevant frames
        attended = self.temporal_attention(encoded[1], encoded[0], encoded[2])
        
        # Metadata attention: use QP/Depth to modulate
        enhanced = self.metadata_attention(attended, metadata)
        
        # Main processing
        out = self.blocks(enhanced)
        
        # Output residual
        residual = self.output_conv(out)
        
        return frames[1] + residual  # Add to current frame
```

### Step 4: Update Dataset

**File: `enhancer/dataset_blackfyre.py`**

```python
# Enhanced feature order
FEATURE_ORDER = [
    # Frame info (4)
    "QP", "PredMode", "Depth", "Boundary",
    # Motion vectors (4)
    "MVL0_X", "MVL0_Y", "MVL1_X", "MVL1_Y",
    # NEW: Additional metadata (8)
    "QT_Depth", "BT_Depth",
    "SkipFlag", "MergeFlag",
    "InterDir", "RefIdxL0", "RefIdxL1",
    "Cbf_Y",
]
# Total: 16 channels
```

### Step 5: Training Script

**File: `train_blackfyre.py`**

Same loss function as Hightower:
- 0.1 * MS-SSIM
- 0.1 * SSIM
- 0.5 * L1
- 0.3 * L2

Plus wandb image logging (in color, every 10 epochs).

---

## Expected Results

| Method | PSNR Gain |
|--------|-----------|
| Hightower (concat) | +0.33 dB |
| **Blackfyre (Temporal Attention + Enhanced Metadata)** | **+0.4 - 0.6 dB** |

---

## Implementation Checklist

- [ ] Update `features_parser/parser.py` to parse additional metadata
- [ ] Update `features_generator/generator.py` to generate more channels
- [ ] Update precompute script to include new metadata
- [ ] Create `enhancer/models/blackfyre.py` with Temporal Attention
- [ ] Create `enhancer/dataset_blackfyre.py` with enhanced features
- [ ] Create `train_blackfyre.py` training script
- [ ] Add wandb color image logging to training
- [ ] Test and train
- [ ] Compare results with Hightower

---

## Why This Approach is Better

1. **Uses Decoder Information**: We have rich metadata from VVC - use it!
2. **No Optical Flow Needed**: Decoder MVs are already optimized for VVC
3. **Learnable Attention**: Network learns what to focus on
4. **Metadata-Guided**: QP tells us where quality is worse
5. **SOTA-Inspired**: Based on papers that use decoder information

---

## References

1. Fatemeh Nasiri et al., "Prediction-Aware Quality Enhancement of VVC Using CNN", 2021
2. "Multi-Frame Quality Enhancement Model for VVC", IEEE 2024
3. "Attention-Based Dual-Scale CNN In-Loop Filter for VVC", IEEE 2019
4. "DREFNet: Deep Residual Enhanced Feature GAN for VVC", MDPI 2023
