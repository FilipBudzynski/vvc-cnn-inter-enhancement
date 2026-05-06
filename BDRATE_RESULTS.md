# BD-Rate / BD-PSNR Results

**Test set (10 unbiased Xiph sequences):** Johnny_1280x720_60, controlled_burn_1080p, pedestrian_area_1080p25, red_kayak_1080p, rush_hour_1080p25, sunflower_1080p25, touchdown_pass_1080p, tractor_1080p25, vidyo1_720p_60fps, vidyo3_720p_60fps
**QPs:** [22, 27, 32, 37, 42]

**Methodology notes:**
- Test set was chosen so no sequence appears in `utils/fetch_dataset.py:SELECTED_VIDEOS` — the training pipeline does a sample-level random split with `seed=42`, so every video in `data/` was seen during training.
- Bitrate from `.vvc` filesize × 8 / num_frames × fps (kbps).
- Y PSNR computed at native resolution. Chroma PSNR uses the same chroma upsample/downsample round-trip used by the model (bilinear up → model → 2×2 avg pool down). An identity-model control across all 50 video-QP pairs showed the round-trip artifact is small and non-systematic: max ±0.07 dB averaged per QP, signed by QP (slightly negative at QP22, slightly positive at QP42), so it does not flip any qualitative conclusion below.
- Bjontegaard BD-PSNR / BD-Rate use the standard cubic-poly formulation (`bdrate.py`).

## BD metrics (lower BD-rate is better; higher BD-PSNR is better)

| Model | Y BD-PSNR (dB) | Y BD-Rate (%) | U BD-PSNR (dB) | U BD-Rate (%) | V BD-PSNR (dB) | V BD-Rate (%) |
|---|---|---|---|---|---|---|
| vvc_ppff | +0.0566 | -1.821 | +0.0311 | -1.593 | +0.0531 | -2.540 |
| martell | +0.1363 | -4.447 | -0.1972 | +7.560 | -0.2124 | +7.092 |
| snow_wide | +0.0948 | -3.210 | -0.2867 | +12.207 | -0.2943 | +10.489 |

## vvc_ppff — per-QP RD points (avg over videos)

| QP | Bitrate (kbps) | Anchor Y/U/V (dB) | Enhanced Y/U/V (dB) | ΔY (dB) | ΔU (dB) | ΔV (dB) |
|---|---|---|---|---|---|---|
| 22 | 6182.9 | 41.72 / 46.25 / 47.32 | 41.68 / 46.17 / 47.22 | -0.035 | -0.082 | -0.101 |
| 27 | 2880.0 | 39.64 / 44.69 / 45.67 | 39.67 / 44.68 / 45.68 | +0.030 | -0.008 | +0.017 |
| 32 | 1456.9 | 37.44 / 43.25 / 44.15 | 37.51 / 43.29 / 44.23 | +0.072 | +0.041 | +0.082 |
| 37 | 706.0 | 35.09 / 41.80 / 42.62 | 35.18 / 41.88 / 42.73 | +0.088 | +0.073 | +0.110 |
| 42 | 313.6 | 32.61 / 40.51 / 41.10 | 32.70 / 40.61 / 41.17 | +0.088 | +0.099 | +0.072 |

## martell — per-QP RD points (avg over videos)

| QP | Bitrate (kbps) | Anchor Y/U/V (dB) | Enhanced Y/U/V (dB) | ΔY (dB) | ΔU (dB) | ΔV (dB) |
|---|---|---|---|---|---|---|
| 22 | 6182.9 | 41.72 / 46.25 / 47.32 | 41.49 / 45.16 / 46.11 | -0.226 | -1.090 | -1.208 |
| 27 | 2880.0 | 39.64 / 44.69 / 45.67 | 39.71 / 44.20 / 45.13 | +0.064 | -0.486 | -0.535 |
| 32 | 1456.9 | 37.44 / 43.25 / 44.15 | 37.65 / 43.16 / 44.05 | +0.209 | -0.087 | -0.100 |
| 37 | 706.0 | 35.09 / 41.80 / 42.62 | 35.33 / 41.94 / 42.79 | +0.242 | +0.136 | +0.168 |
| 42 | 313.6 | 32.61 / 40.51 / 41.10 | 32.82 / 40.73 / 41.36 | +0.214 | +0.220 | +0.263 |

## snow_wide — per-QP RD points (avg over videos)

| QP | Bitrate (kbps) | Anchor Y/U/V (dB) | Enhanced Y/U/V (dB) | ΔY (dB) | ΔU (dB) | ΔV (dB) |
|---|---|---|---|---|---|---|
| 22 | 6182.9 | 41.72 / 46.25 / 47.32 | 41.41 / 45.00 / 45.92 | -0.312 | -1.248 | -1.397 |
| 27 | 2880.0 | 39.64 / 44.69 / 45.67 | 39.64 / 44.08 / 45.01 | -0.004 | -0.605 | -0.660 |
| 32 | 1456.9 | 37.44 / 43.25 / 44.15 | 37.61 / 43.07 / 43.98 | +0.167 | -0.174 | -0.172 |
| 37 | 706.0 | 35.09 / 41.80 / 42.62 | 35.32 / 41.88 / 42.74 | +0.225 | +0.072 | +0.127 |
| 42 | 313.6 | 32.61 / 40.51 / 41.10 | 32.82 / 40.70 / 41.35 | +0.212 | +0.195 | +0.252 |

## Per-video Y BD-Rate (%) by model

| Video | vvc_ppff | martell | snow_wide |
|---|---|---|---|
| Johnny_1280x720_60 | -1.46 | -2.10 | -1.41 |
| controlled_burn_1080p | +0.56 | +1.66 | +2.72 |
| pedestrian_area_1080p25 | -2.85 | -7.66 | -7.18 |
| red_kayak_1080p | -0.30 | -3.50 | -2.89 |
| rush_hour_1080p25 | -2.32 | -8.94 | -8.44 |
| sunflower_1080p25 | -1.66 | -1.68 | -0.32 |
| touchdown_pass_1080p | -1.21 | -2.37 | -0.53 |
| tractor_1080p25 | -2.10 | -4.62 | -3.70 |
| vidyo1_720p_60fps | -1.88 | -1.99 | +0.42 |
| vidyo3_720p_60fps | -2.32 | -3.76 | -1.22 |
