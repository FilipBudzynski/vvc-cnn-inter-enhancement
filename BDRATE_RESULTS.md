# BD-Rate / BD-PSNR Results

**Test set (10 unbiased Xiph sequences):** Johnny_1280x720_60, controlled_burn_1080p, pedestrian_area_1080p25, red_kayak_1080p, rush_hour_1080p25, sunflower_1080p25, touchdown_pass_1080p, tractor_1080p25, vidyo1_720p_60fps, vidyo3_720p_60fps
**QPs:** [22, 27, 32, 37, 42]

## BD metrics (lower BD-rate is better; higher BD-PSNR is better)

| Model | Y BD-PSNR (dB) | Y BD-Rate (%) | U BD-PSNR (dB) | U BD-Rate (%) | V BD-PSNR (dB) | V BD-Rate (%) |
|---|---|---|---|---|---|---|
| vvc_ppff | +0.0566 | -1.821 | +0.0311 | -1.593 | +0.0531 | -2.540 |
| martell | +0.1363 | -4.447 | -0.1972 | +7.560 | -0.2124 | +7.092 |

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

## Per-video Y BD-Rate (%) by model

| Video | vvc_ppff | martell |
|---|---|---|
| Johnny_1280x720_60 | -1.46 | -2.10 |
| controlled_burn_1080p | +0.56 | +1.66 |
| pedestrian_area_1080p25 | -2.85 | -7.66 |
| red_kayak_1080p | -0.30 | -3.50 |
| rush_hour_1080p25 | -2.32 | -8.94 |
| sunflower_1080p25 | -1.66 | -1.68 |
| touchdown_pass_1080p | -1.21 | -2.37 |
| tractor_1080p25 | -2.10 | -4.62 |
| vidyo1_720p_60fps | -1.88 | -1.99 |
| vidyo3_720p_60fps | -2.32 | -3.76 |
