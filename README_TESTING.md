# Martell Model Testing Guide

## Important: Use SEPARATE Test Data!

### DO NOT use training data for evaluation!
- Training data = `data/precomputed_martell/` (already used during training)
- Test data = separate videos NOT used in training

## Steps to Generate Test Data

### 1. Prepare Test Videos
```bash
# Create directory for test videos (NOT in training set!)
mkdir -p test_videos

# Copy test videos here (different from training!)
# Example: Different video sequences not used in training
```

### 2. Compress with VVC at Multiple QPs
```bash
# Compress test videos with VTM at QP 22, 27, 32, 37, 42
# Save compressed outputs to:
# test_videos/vtm_output/qp22/
# test_videos/vtm_output/qp27/
# etc.
```

### 3. Precompute Features for Test Data
```bash
# Use the precompute script (same as training but on test videos!)
python scripts/precompute_features.py \
    --input-dir test_videos/vtm_output/qp22 \
    --output-dir data_qp22_test \
    --model martell

# Repeat for other QPs:
# data_qp27_test, data_qp32_test, data_qp37_test, data_qp42_test
```

### 4. Run Evaluation
```bash
# Activate environment
source .venv/bin/activate

# Run evaluation (calculates BD-rate for Y, U, V)
python evaluate_martell_bdrate.py
```

## Expected Output

The evaluation will produce:
1. **Per-QP results** for Y, U, V channels
2. **Average PSNR gain** for each channel
3. **BD-rate** (Bjøntegaard Delta Rate) for each channel
   - Negative BD-rate = GOOD (bitrate savings)
   - Example: -7.34% means 7.34% bitrate savings at same quality

## BD-Rate Interpretation

| BD-Rate (Y) | Meaning |
|--------------|---------|
| -10% to -5% | Excellent improvement |
| -5% to -2% | Good improvement |
| -2% to 0% | Moderate improvement |
| 0% to +2% | No significant change |
| > +2% | Worse than anchor |

## Files Created
- `martell_bdrate_results.json` - Detailed results
- `martell_bdrate.log` - Evaluation log

## Comparison with Other Models

Compare Martell BD-rate with:
- **Snow-Wide** (baseline)
- **STENet (2024)** - when implemented
- **VVC-PPFF** - if available

## Paper Citation

When reporting results, cite:
```
Martell model (Snow-Wide + metadata):
- Trained on VVC-compressed video (Random Access config)
- Uses 3 frames (prev, curr, next) + 9-dim metadata
- BD-rate (Y): XX.XX%
- BD-rate (U): XX.XX%
- BD-rate (V): XX.XX%
```
