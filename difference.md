# Differences Between Current Implementation and Piotr's Work

This document compares the current repository (VVC CNN Inter-Enhancement) with Piotr's original work (VVC GAN Decode Enhancement).

---

## 1. Task Type

| Aspect | Piotr's Work | Current Repo |
|--------|--------------|--------------|
| **Encoding Type** | Inter (random access) + Intra | **Intra only** |
| **Network Type** | GAN (Generator + Discriminator) | **CNN only** (no GAN) |
| **Goal** | Perceptual quality (SSIM, visual appeal) | PSNR-focused enhancement |

---

## 2. Architecture Comparison

### 2.1 Input Processing (CRITICAL DIFFERENCE)

**Piotr's Approach:**
```python
# enhancer.py - Lines 68-72
def forward(self, input_: Tensor, metadata: Tensor) -> Tensor:
    shape = input_.shape[2:]
    encoded_metadata = self.metadata_encoder(metadata, shape)  # Interpolates metadata!
    data = torch.cat((input_, encoded_metadata), 1)  # Concat at channel dim
    result = self.model(data)
```

**Key insight:** Piotr uses a **MetadataEncoder** that:
1. Takes metadata as separate tensor `[B, 6, 1, 1]` (6 scalar features per sample)
2. Interpolates it to match input size using `torch.nn.functional.interpolate`
3. Then concatenates with the 3-channel input

**Current Repo:**
- Metadata is already expanded to feature maps (8 channels of 128x128)
- Concatenated directly: `[B, 11, 128, 128]` (3 YUV + 8 metadata maps)

### 2.2 Metadata Encoding

| Feature | Piotr's Work | Current Repo |
|---------|--------------|--------------|
| Metadata Shape | `[B, 6]` scalars → interpolated to `[B, 6, H, W]` | `[B, 8, H, W]` full feature maps |
| QP | `qp / 64` (scalar) | `t / 63.0` (full map) |
| Profile | RA=0, AI=1 (scalar) | N/A (not in current) |
| ALF | 0/1 (scalar) | N/A (not in current) |
| SAO | 0/1 (scalar) | N/A (not in current) |
| DB | 0/1 (scalar) | N/A (not in current) |
| Depth | N/A | Full map |
| PredMode | N/A | Full map |
| Boundary | N/A | Full map |
| Motion Vectors | N/A | Full maps (4 channels) |

### 2.3 Model Architectures

**Best performer in Piotr's work (from Experiments.md):**
- **DenseNet** with structure: 64 → 96 → 64 → 48 → 32 channels
- kernel sizes: 9 → 7 → 5 → 3 → 3
- Features: 64 channels with dense connections
- **val_psnr: 36.10 dB**, val_ssim: 0.963

**Current config (res.yaml):**
- ResNet with 128 channels, 5 blocks, 3 layers each
- kernel_size: 3 for all blocks
- No dense connections

---

## 3. Training & Loss Functions

### 3.1 Loss Functions

**Piotr's Combined Loss (trainer_module.py, line 160):**
```python
g_loss = 0.1 * msssim_loss + 0.1 * ssim_loss + mse_loss + 0.5 * l1_loss
```

Weights:
- MS-SSIM: 0.1
- SSIM: 0.1  
- MSE: 1.0
- L1: 0.5

**Current Implementation:**
- Charbonnier Loss (robust L1): 1.0 for Y, 0.5 for U/V
- SSIM Loss: 0.1 (recently added)
- No MS-SSIM loss

### 3.2 Channel Gradients

**Piotr's approach (dense.yaml, lines 4-7):**
```yaml
channels_grad_scales:
  - 0.66666
  - 0.66666
  - 0.66666
```

Equal weighting for all Y, U, V channels.

**Current Implementation:**
```python
total_loss = (1.0 * loss_Y) + (0.5 * loss_U) + (0.5 * loss_V)
```

Different weighting (Y is more important).

### 3.3 Optimizer

| Aspect | Piotr's Work | Current Repo |
|--------|--------------|--------------|
| Generator | Adam (lr=0.0002, betas=(0.5, 0.999)) | Adam (lr=0.0001) |
| Discriminator | SGD (lr=0.0001, momentum=0.9) | N/A |
| Scheduler | MultiStepLR (milestones: 50,100,150,200,300,400,500,600,800) | Linear warmup + Cosine |

---

## 4. Data Processing

### 4.1 Data Format

**Piotr's Work:**
- Chunks saved as PNG files (132x132)
- Separate folders for decoded and original chunks
- Metadata stored in folder structure/file names

**Current Repo:**
- YUV files (4:2:0 format)
- Metadata generated from VTM trace CSV
- On-the-fly processing with random crops

### 4.2 Normalization

**Piotr's (dataset.py, lines 200-209):**
```python
def _metadata_to_np(self, metadata: Metadata) -> Any:
    return np.array((
        0 if metadata.profile == "RA" else 1,  # profile
        metadata.qp / 64,                        # QP normalized to [0,1]
        metadata.alf,                            # 0 or 1
        metadata.sao,                            # 0 or 1
        metadata.db,                             # 0 or 1
        metadata.is_intra,                       # 0 or 1
    ))
```

**Current Repo (vtm_dataset.py):**
```python
if name == "QP":
    return (t / 63.0).clamp(0, 1)
elif name == "Depth":
    return (t / 7.0).clamp(0, 1)
elif name == "PredMode":
    return (t / 3.0).clamp(0, 1)
elif "MV" in name:
    return torch.tanh(t / 64.0)  # ~[-1, 1]
```

---

## 5. Key Findings from Piotr's Experiments

### Best Results (from Experiments.md):

| Model | val_psnr | val_ssim | Notes |
|-------|----------|----------|-------|
| DenseNet (1000 epochs) | **36.10** | 0.963 | Best overall |
| ResNet (1000 epochs) | 35.60 | 0.951 | |
| DenseNet (98 epochs) | 36.16 | 0.951 | Earlier checkpoint |
| Dense (no mask) | 30.24 | 0.929 | Without residual learning |

### Critical Observations:

1. **Without mask (no residual learning)**: PSNR drops from 36 to 30 dB!
2. **DenseNet > ResNet** in Piotr's experiments
3. **More epochs help**: 998 epochs gave better results than 98 epochs

---

## 6. Recommendations for Current Repo

### 6.1 Architecture Changes

1. **Try DenseNet instead of ResNet** - It performed better in Piotr's work
2. **Use progressive channel reduction**: 64 → 96 → 64 → 48 → 32 (like Piotr's best config)
3. **Use larger kernel in first layer**: 9x9 instead of 7x7

### 6.2 Metadata Handling

**Option A - Keep full metadata maps (current approach):**
- Fix normalization to be consistent ([0,1] for all channels)
- Better for motion vectors and spatial features

**Option B - Use Piotr's approach (simpler metadata):**
- Reduce metadata to 6 scalars per sample
- Interpolate to match spatial resolution
- Less memory, potentially faster training

### 6.3 Training Improvements

1. **Add MS-SSIM loss** (Piotr used this)
2. **Adjust loss weights**: `0.1 * msssim + 0.1 * ssim + mse + 0.5 * l1`
3. **Use MultiStepLR scheduler** (more milestones)
4. **Train longer** - Piotr used 1000 epochs

### 6.4 Data Augmentation

Piotr's work doesn't explicitly mention augmentation, but adding:
- Random horizontal/vertical flips
- Channel swapping (U↔V for chroma robustness)

---

## 7. Summary of Critical Differences

| Aspect | Piotr's | Current | Action |
|--------|---------|---------|--------|
| Task | Inter encoding | Intra encoding | N/A |
| Network | GAN | CNN only | Keep CNN |
| Model | DenseNet 64→96→64→48→32 | ResNet 128x5 | Try DenseNet |
| Metadata | 6 scalars, interpolated | 8 full maps | Keep full maps |
| Loss | MS-SSIM+SSIM+MSE+L1 | Charbonnier+SSIM | Add MS-SSIM |
| Epochs | 1000 | 200 | Increase |
| Mask/Residual | Critical (30→36 dB) | with_mask=true | Keep |
| LR Schedule | MultiStepLR | Linear+Cosine | Try MultiStep |

---

## 8. Conclusion

The current repository has the **foundation** (residual learning, metadata concatenation) but needs:

1. **Better loss function** (add MS-SSIM)
2. **DenseNet architecture** instead of ResNet
3. **More training epochs** (1000 vs 200)
4. **Fix metadata normalization** (consistent [0,1] range)

The key insight from Piotr's work: **Residual learning (with_mask) is critical** - without it, performance drops by 6 dB!

---
