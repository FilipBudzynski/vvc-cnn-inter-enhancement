# Improvements for VVC CNN Inter-Enhancement

This document outlines all identified issues, potential improvements, and recommendations for the repository.

---

## Table of Contents
1. [Critical Issues](#critical-issues)
2. [Data Preprocessing & Normalization](#data-preprocessing--normalization)
3. [Model Architecture](#model-architecture)
4. [Adopt from Piotr's Work](#adopt-from-piotrs-work)
5. [Training & Loss](#training--loss)
6. [Code Quality](#code-quality)
7. [Recommendations Summary](#recommendations-summary)

---

## Critical Issues

### 1. PSNR Calculation Uses Wrong Data Range
**File:** `enhancer/trainer.py` (line 134-136)

```python
# Current (WRONG):
t_psnr_y = psnr(eY * 255.0, oY * 255.0, data_range=255.0)

# Data is already normalized to [0, 1], so this should be:
t_psnr_y = psnr(eY, oY, data_range=1.0)
```

**Impact:** Test metrics are being calculated incorrectly. This explains confusion about results.

---

### 2. Zero-Metadata Performance is BETTER Than Real Metadata
**Symptom:** `test_zero_meta_psnr_Y: 32.47` > `test_psnr_Y: 32.38`

This indicates the model learns to USE metadata incorrectly - it performs better when metadata is zeroed out!

**Root Causes:**
1. **Inconsistent normalization ranges**: MV channels use `tanh()` (~[-1, 1]), while other metadata is [0, 1]
2. **Model may not be learning to leverage metadata effectively**
3. **The `Enhancer` class was not being used before** (now fixed)

---

## Data Preprocessing & Normalization

### 3. Inconsistent Metadata Value Ranges
**File:** `enhancer/vtm_dataset.py` (lines 97-110)

| Channel | Current Range | Issue |
|---------|---------------|-------|
| QP | [0, 1] | OK |
| Depth | [0, 1] | OK |
| PredMode | [0, 1] | OK |
| Boundary | raw (0 or 1) | Not explicitly normalized |
| MV (all) | ~[-1, 1] via tanh | **Different range!** |

**Recommendation:** Normalize ALL metadata to [0, 1] for consistency:

```python
def _normalize_metadata(self, name: str, data: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(data).float()
    
    if name == "QP":
        return (t / 63.0).clamp(0, 1)
    elif name == "Depth":
        return (t / 7.0).clamp(0, 1)
    elif name == "PredMode":
        return (t / 3.0).clamp(0, 1)
    elif "MV" in name:
        # Normalize to [0, 1] instead of tanh
        return (t / 128.0).clamp(-1, 1)  # MV range typically [-128, 127]
    elif name == "Boundary":
        return (t / 1.0).clamp(0, 1)  # Explicit normalization
    
    return t
```

---

### 4. No Data Augmentation
**File:** `enhancer/vtm_dataset.py`

Currently only applies:
- Random cropping (aligned to 8-pixel grid)
- Padding

**Missing augmentations that could help:**
- Horizontal/vertical flipping
- Channel swapping (U↔V for chroma robustness)
- Random brightness/contrast adjustments
- CutMix or MixUp for regularization

**Recommendation:** Add augmentations in `__getitem__`:

```python
# After cropping, before returning:
if random.random() > 0.5:
    x = torch.flip(x, dims=[-1])  # Horizontal flip
    original_patch = torch.flip(original_patch, dims=[-1])
```

---

### 5. Train/Val/Test Split is Fixed (No Stratification)
**File:** `enhancer/datamodule.py` (lines 27-32)

```python
random.seed(42)
random.shuffle(all_videos)
n = len(all_videos)
train_end = int(n * 0.8)
validate_end = int(n * 0.9)
```

**Issues:**
- No stratification by video content type (e.g., high motion vs. static)
- 80/10/10 split may not be optimal
- Fixed random seed makes it non-reproducible across runs without seed

---

### 6. Video-Level Split May Cause Distribution Shift
**Issue:** Entire videos go into one split (train/val/test), not frames.

This is actually CORRECT behavior, but the small test set (10%) may not be representative.

---

## Model Architecture

### 7. Small Model Capacity (Already Addressed)
**File:** `config.yaml`

Changed from 64 to 128 channels, 2 to 3 layers per block.

**Further consideration:**
- Try 256 channels for larger model
- Add attention mechanisms (CBAM, SE blocks)

---

### 8. No Skip Connections in Feature Processing
**Observation:** The `Enhancer` class uses residual learning (`with_mask`), but there's no intermediate skip connections.

**Recommendation:** Consider adding UNet-style skip connections for better gradient flow.

---

### 9. Output Layer Initialized to Zero
**File:** `enhancer/models/conv.py` (lines 148-152)

```python
nn.init.zeros_(conv_module.weight)
```

This is intentional for residual learning but may slow convergence.

**Alternative:** Use Kaiming initialization for faster convergence:
```python
nn.init.kaiming_normal_(conv_module.weight, mode='fan_out', nonlinearity='relu')
```

---

## Adopt from Piotr's Work

Based on comparison with Piotr's VVC GAN Decode Enhancement repository (see `difference.md`).

### 10. Switch to DenseNet Architecture
**Evidence:** Piotr's best results used DenseNet with:
- Channels: 64 → 96 → 64 → 48 → 32 (progressive reduction)
- Kernel sizes: 9 → 7 → 5 → 3 → 3 (progressive reduction)
- val_psnr: **36.10 dB**, val_ssim: **0.963**

**Current:** ResNet with uniform 128 channels, 3x3 kernels

**Recommendation:** Use DenseNet with progressive channel reduction like Piotr's best config:

```yaml
enhancer:
  implementation: "dense"  # Change from "res"
  features:
    kernel_size: 9        # Larger kernel
    padding: 4
    stride: 1
    features: 64
    pool: false
    dense: true           # Enable dense connections in features
  structure:
    blocks:
      - { num_layers: 4, features: 64, kernel_size: 7, transition: { mode: same } }
      - { num_layers: 4, features: 96, kernel_size: 5, transition: { mode: same } }
      - { num_layers: 4, features: 64, kernel_size: 3, transition: { mode: same } }
      - { num_layers: 4, features: 48, kernel_size: 3, transition: { mode: same } }
      - { num_layers: 4, features: 32, kernel_size: 3 }
```

---

### 11. Add MS-SSIM Loss
**Evidence:** Piotr's loss function:
```python
g_loss = 0.1 * msssim_loss + 0.1 * ssim_loss + mse_loss + 0.5 * l1_loss
```

**Current:** Only Charbonnier + SSIM

**Recommendation:** Add MS-SSIM (Multi-Scale SSIM) loss from `enhancer/ssim.py`:

```python
from enhancer.ssim import MS_SSIM

self.msssim = MS_SSIM(data_range=1.0, win_size=9, per_channel=True)

# In loss calculation:
msssim_loss = 1 - self.msssim(target, prediction)
total_loss = 0.1 * msssim_loss + 0.1 * ssim_loss + mse_loss + 0.5 * l1_loss
```

---

### 12. Use MultiStepLR Scheduler
**Evidence:** Piotr used milestones at [50, 100, 150, 200, 300, 400, 500, 600, 800]

**Current:** Linear warmup + Cosine annealing

**Recommendation:** Use MultiStepLR with more milestones:

```python
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[50, 100, 150, 200, 300, 400, 500, 600, 800],
    gamma=0.5
)
```

---

### 13. Train for More Epochs
**Evidence:** Piotr's best DenseNet ran for **1000 epochs**
- val_psnr at epoch 98: 36.16
- val_psnr at epoch 998: 36.10 (sustained)

**Current:** 200 epochs

**Recommendation:** Increase to 500-1000 epochs.

---

### 14. Use MetadataEncoder Approach
**Evidence:** Piotr used a separate metadata encoder that:
1. Takes 6 scalar metadata values per sample
2. Interpolates to spatial resolution
3. Concatenates with input

**Current:** Full feature maps for all metadata

**Recommendation:** This is optional - current approach with full maps is actually MORE detailed. Keep current approach but ensure consistent normalization.

---

### 15. Channel Gradient Scaling
**Evidence:** Piotr used equal weighting for all channels:
```yaml
channels_grad_scales:
  - 0.66666
  - 0.66666
  - 0.66666
```

**Current:** Y=1.0, U=0.5, V=0.5

**Recommendation:** Consider equal weighting or adjust based on experiments.

---

## Training & Loss

### 16. Loss Function Now Includes SSIM (Already Fixed)
**File:** `enhancer/trainer.py`

Added SSIM loss with 10% weight. This was a good improvement.

**Further consideration:**
- Try adding MS-SSIM (multi-scale SSIM) - see section 11
- Consider perceptual loss (VGG-based) for better visual quality

---

### 17. Learning Rate Schedule
**File:** `enhancer/trainer.py` (lines 188-209)

Currently uses:
- Linear warmup (5 epochs)
- Cosine annealing

**Consider:**
- Using MultiStepLR like Piotr - see section 12
- Adding warm restarts
- Using OneCycleLR for faster convergence

---

### 18. No Gradient Clipping
**Recommendation:** Add gradient clipping to prevent exploding gradients:

```python
# In trainer.py configure_optimizers:
return {
    "optimizer": optimizer,
    "lr_scheduler": {...},
    "gradient_clip_val": 1.0,  # Add this
}
```

---

## Code Quality

### 19. Hardcoded Paths
**File:** `enhancer/datamodule.py` (line 45)

```python
orig_dir = Path("data")  # Hardcoded!
```

**Recommendation:** Make configurable via config.yaml.

---

### 20. Magic Numbers Throughout Code
**Examples:**
- `patch_size: 128` - should be in config
- `batch_size: 16` - should be in config
- `num_workers: 4` - should be in config

---

### 21. No Input Validation
**Issue:** No validation that decoded YUV and original YUV have the same dimensions.

---

### 22. Dataset Error Handling
**File:** `enhancer/vtm_dataset.py` (lines 126-141)

Returns zeros for missing frames instead of raising errors. This could lead to silent failures.

---

### 23. No Logging of Dataset Statistics
**Recommendation:** Log mean/std of dataset for debugging:

```python
def setup(self, stage=None):
    # After loading dataset:
    print(f"Dataset mean: {dataset.mean()}, std: {dataset.std()}")
```

---

## Recommendations Summary

### High Priority (Fix Now)
1. **Fix PSNR calculation** - use `data_range=1.0`
2. **Fix metadata normalization** - consistent [0,1] range
3. **Verify data alignment** - ensure original/decoded frames are aligned
4. **Switch to DenseNet** - use Piotr's proven architecture
5. **Add MS-SSIM loss** - critical for visual quality

### Medium Priority (Next Iteration)
6. Add data augmentation
7. Increase epochs to 500+
8. Use MultiStepLR scheduler
9. Add gradient clipping
10. Add skip connections / attention

### Lower Priority (Future)
11. Add more augmentations (MixUp, CutMix)
12. Try perceptual loss
13. Implement test-time augmentation (TTA)
14. Add input validation and error handling
15. Make paths configurable

---

## Testing Checklist

After implementing fixes, verify:

- [ ] `test_gain_Y` is POSITIVE (model improves over VVC)
- [ ] `test_psnr_Y` > `test_zero_meta_psnr_Y` (metadata helps)
- [ ] `test_psnr_Y` > `test_ref_psnr_Y` (model improves over input)
- [ ] Validation and test metrics are similar (no distribution shift)

---

*Generated: 2025*
