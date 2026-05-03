# STENet Implementation Plan (2024)
# Based on: "Joint Reference Frame Synthesis and Post Filter Enhancement for Versatile Video Coding"
# arXiv: 2404.18058 | IEEE 2024

## Overview
- **Most fresh VVC-specific CNN paper (2024)**
- **No existing implementation** (PapersWithCode: "No code implementations yet")
- **BD-rate claimed**: -7.34% (Y), -17.21% (U), -16.65% (V) under RA config

## Architecture: Space-Time Enhancement Network (STENet)

### Input/Output:
```
Input: Two reconstructed frames I₀, I₁ (from VVC decoder)
Output: 
  - I_Enh₀, I_Enh₁ (enhanced frames for PFE)
  - I_Syn^t (synthesized intermediate frame for RFS)
```

### Two Pipelines:
1. **Enhancement Pipeline (PFE)**: Enhances both input frames
2. **Synthesis Pipeline (RFS)**: Synthesizes intermediate virtual reference frame

### Key Components:
- Optical Flow Estimation (IFRNet-based, fast inference)
- Bidirectional recurrent architecture
- Joint Inference of RFS and PFE (JISE) - single execution

## Adaptation for Martell Data (Decoder Output)

### Key Difference:
| Aspect | Martell (Your Model) | STENet (Paper) |
|---------|---------------------|-------------------|
| **Data Source** | ✅ DECODER (metadata: QP, frame type, motion vectors) | DPB (pixel values only) |
| **Input** | 3 frames (prev, curr, next) + metadata | 2 frames (pixel values only) |
| **Advantage** | Richer metadata for quality-aware processing | Simpler, but less informed |

### Modifications for Your Data:
1. **Use 2 frames instead of 3** (match STENet)
2. **Integrate metadata** (QP, frame type as additional channels) - YOUR ADVANTAGE
3. **Skip optical flow** (use motion vectors from your metadata)
4. **Train on same `data/precomputed_martell/`**

## Implementation Steps

### Phase 1: Core STENet Architecture
```python
# enhancer/models/stenet.py
class STENet(nn.Module):
    """
    Space-Time Enhancement Network
    Input: I0, I1 (batch, 2*3, H, W) if concatentated
    Output: I_Enh0, I_Syn, I_Enh1
    """
    def __init__(self, base_channels=64):
        # Optical flow estimation (or use metadata motion vectors)
        # Synthesis pipeline (for RFS)
        # Enhancement pipeline (for PFE)
        # Joint inference capability
        pass
```

### Phase 2: Data Preparation
```python
# scripts/precompute_stenet.py
# Modify to create 2-frame sequences (not 3)
# Keep metadata for potential use
```

### Phase 3: Training Strategy (from Paper)
```python
# train_stenet.py
# Joint training: Loss = α*Loss_PFE + β*Loss_RFS
# Use same data as Martell for fair comparison
```

### Phase 4: Evaluation
```python
# evaluate_stenet.py
# Compare with Martell on same VVC sequences
# BD-rate for Y, U, V (like Martell evaluation)
```

## Files to Create:

```
enhancer/models/stenet.py           # STENet architecture
enhancer/models/optical_flow.py      # Optical flow (or use metadata)
scripts/precompute_stenet.py         # Data preparation (2-frame)
train_stenet.py                      # Training script
evaluate_stenet.py                  # Evaluation script
```

## Comparison Target:
- **Martell model** (Snow-Wide + metadata + 3 frames)
- Train STENet on **same data** for fair comparison
- **Expected**: STENet should outperform Martell (paper claims -7.34% BD-rate)

## References:
- Paper: https://arxiv.org/abs/2404.18058
- VTM Integration: VTM-15.0
- STEW: Space-Time Enhancement Window (groups of 8 frames)
- JISE: Joint Inference of RFS and PFE
