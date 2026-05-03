# LSTM-Based Video Enhancement Model Plan

## Research Summary

Based on recent papers (2019-2025) on LSTM for video compression artifact removal:

### Key Papers:
1. **QG-ConvLSTM** (Yang et al., ICME 2019)
   - Quality-Gated Convolutional LSTM with bi-directional structure
   - Learns "forget" and "input" gates from quality-related features
   - Frames with different quality contribute differently to memory

2. **NL-ConvLSTM** (Xu et al., ICCV 2019)
   - Non-local ConvLSTM for capturing global motion patterns
   - Approximate non-local strategy for efficiency
   - Uses preceding and following frames without explicit motion compensation

3. **MFQE 2.0** (2019)
   - Uses BiLSTM for Peak Quality Frame (PQF) detection
   - Enhances non-PQF frames using neighboring PQFs
   - Motion compensation subnet + Quality enhancement subnet

4. **MGANet** (2019)
   - Bidirectional Residual ConvLSTM (BRCLSTM)
   - Implicitly discovers inter-frame variations
   - Multi-scale guided encoder-decoder with skip connections

## Proposed Architecture: Quality-Aware ConvLSTM (QA-ConvLSTM)

### Design Goals:
- Leverage existing metadata (QP, frame type, bits, motion vectors)
- Use bi-directional ConvLSTM to process temporal information
- Incorporate quality-gating mechanism for smart frame weighting
- Maintain compatibility with existing preprocessing pipeline

### Architecture Components:

```
1. Feature Extraction (from Snow-Wide)
   - Input: Current frame + prev + next (3 channels each = 9 channels)
   - Metadata: 9-dimensional quality-related features
   - Output: Feature maps (64 channels)

2. Bi-Directional ConvLSTM
   - Forward LSTM: Processes frames 1...t
   - Backward LSTM: Processes frames t...1
   - Quality-Gated Gates:
     * Forget gate: f_t = σ(W_f * [h_{t-1}, x_t] + bias_f) * quality_weight
     * Input gate: i_t = σ(W_i * [h_{t-1}, x_t] + bias_i) * quality_weight
   - Quality weights derived from: QP, frame type, bitrate, PSNR

3. Temporal Attention Fusion
   - Attention mechanism to focus on high-quality neighboring frames
   - Non-local module (optional) for long-range dependencies

4. Reconstruction Decoder
   - Deep decoder with skip connections (from Snow-Wide)
   - Output: Enhanced frame (3 channels)

5. Loss Function (from Martell)
   - 0.5*L1 + 0.15*MS-SSIM + 0.2*GradLoss + 0.15*Laplacian
```

### Input Features (Quality-Gating):
```python
quality_features = [
    QP,                    # Quantization parameter
    frame_type_I,           # I-frame (1) or not (0)
    frame_type_P,           # P-frame
    frame_type_B,           # B-frame
    bits_per_pixel,         # Compression ratio indicator
    temporal_layer,         # For hierarchical B-frames
    motion_intensity,       # From motion vectors
    psnr_estimate,          # No-reference quality estimate
    bitrate,                # Total bits for frame
]
```

### Data Preparation:
- Extend `precompute_features.py` to extract:
  - Multi-frame sequences (current + 3 previous + 3 next frames)
  - Quality features for each frame
  - Optical flow or motion vectors (optional)

### Implementation Steps:

#### Phase 1: Basic ConvLSTM
1. Implement ConvLSTM cell (`enhancer/models/conv_lstm.py`)
2. Create bidirectional wrapper
3. Simple integration with existing feature extraction
4. Train on single-frame + temporal context

#### Phase 2: Quality-Gating
1. Add quality feature extraction to `features_parser`
2. Implement quality-gated gates in ConvLSTM
3. Train with quality-aware weighting

#### Phase 3: Advanced Features
1. Add non-local attention (optional)
2. Multi-frame sequence training
3. Compare with Snow-Wide baseline

### Files to Create/Modify:

```
enhancer/models/
  ├── conv_lstm.py           # ConvLSTM cell + Quality-Gated variant
  ├── qa_conv_lstm.py       # Quality-Aware Bi-Directional ConvLSTM model
  └── (modify) snow_wide.py # Optional: hybrid approach

features_parser/
  └── (modify) parser.py    # Extract quality features

scripts/
  └── precompute_lstm.py    # Generate multi-frame sequences

train_lstm.py               # Training script
evaluate_lstm.py            # Evaluation script
```

### Comparison with Baseline (Snow-Wide):
- **Snow-Wide**: 3 frames (prev, curr, next) + metadata, no temporal memory
- **QA-ConvLSTM**: Multiple frames + quality-gated temporal memory

### Expected Improvements:
- Better temporal consistency (less flickering)
- Smarter use of high-quality frames
- Ability to leverage long-range temporal dependencies
- Quality-adaptive enhancement

### Evaluation Metrics:
- PSNR, SSIM, MS-SSIM
- Temporal consistency (PSNR/std across frames)
- BD-Rate analysis
- Visual quality (artifact reduction)

## Timeline:
1. Week 1: Implement ConvLSTM cell + basic training
2. Week 2: Add quality-gating mechanism
3. Week 3: Multi-frame sequence training + evaluation
4. Week 4: Optimization + comparison with Snow-Wide

## References:
- QG-ConvLSTM: https://arxiv.org/abs/1903.04596
- NL-ConvLSTM: https://arxiv.org/abs/1910.12286
- MFQE 2.0: https://arxiv.org/abs/1902.09707
- MGANet: http://arxiv.org/pdf/1811.09150v1
