# VVC CNN Inter-frame Enhancement — BD-rate evaluation pipeline

Master's thesis: post-processing CNNs for VVC-decoded video, evaluated against published paper baselines on a fully unbiased test set.

This branch (`bdrate-eval`) holds:
- **One canonical evaluation pipeline**.
- **All BD-rate / BD-PSNR results** measured on a 10-video Xiph test set with no overlap with the training pool.
- **Training drivers** for every model in the leaderboard.

## Final leaderboard

10 unbiased Xiph sequences (Johnny, vidyo1, vidyo3 (720p) + 7×1080p), QPs {22, 27, 32, 37, 42}, full Bjontegaard cubic. Negative BD-Rate = improvement.

| Model | Y BD-Rate | U BD-Rate | V BD-Rate | Params | Loss |
|---|---:|---:|---:|---:|---|
| VVC-PPFF *(paper, Appl. Sci. 2024)* | −1.82 % | −1.59 % | −2.54 % | 2.78 M | MSE |
| STENet *(paper, arXiv 2404.18058)* | 0.00 % | −1.91 % | −1.95 % | 761 K | MSE + 0.1·MSE(synth) |
| Bi-ConvLSTM *(QG-ConvLSTM baseline, ICME 2019)* | 0.00 % | −1.26 % | −1.68 % | 364 K | MSE |
| Martell *(thesis, original loss)* | **−4.45 %** | +7.56 % ⚠ | +7.09 % ⚠ | 1.29 M | L1 + MS-SSIM + Sobel + Laplacian |
| Snow-Wide *(thesis, retrained 9-ch)* | −3.21 % | +12.21 % ⚠ | +10.49 % ⚠ | 1.29 M | same as original Martell |
| Martell-MSE *(this work)* | −2.28 % | −2.10 % | −1.90 % | 1.29 M | MSE |
| **Martell-Hybrid** ⭐ *(this work)* | **−3.06 %** | **−1.39 %** | **−1.83 %** | 1.29 M | original loss on Y + MSE on chroma |
| Martell-Hybrid-FT | −3.21 % | +1.18 % ⚠ | +0.07 % | 1.29 M | FT @ chroma_w=50 (overfit) |

⚠ = chroma regression. Recommended checkpoint: **`checkpoints/martell_hybrid_best.pt`**.

Full per-QP and per-video breakdown in [`BDRATE_RESULTS.md`](BDRATE_RESULTS.md). Training-loss curves in W&B project `vvc-cnn-inter`.

## Pipeline

```
fetch_eval_videos.py     # download 10 unbiased Xiph y4m → 64-frame YUV + .info
prepare_eval.py          # vvenc-encode and VTM-decode-with-traces at 5 QPs
evaluate_bd.py           # run a model on every (prev, curr, next) triplet,
                         # compute Y/U/V PSNR before/after, compute Bjontegaard
summarize_bd.py          # produce comparison markdown across multiple models
sync_to_wandb.py         # post-hoc upload of training-log metrics to W&B
```

Quick run-through:
```bash
python fetch_eval_videos.py
python prepare_eval.py
python evaluate_bd.py --model martell --checkpoint checkpoints/martell_hybrid_best.pt \
    --out bdrate_results/martell.json
python summarize_bd.py bdrate_results/*.json -o BDRATE_RESULTS.md
```

## Training a new model

The training drivers all share the same recipe (Adam(lr=1e-4, wd=1e-4), MultiStepLR `[50, 100, 150, 200, 300]`, batch 8, patch 132, 200 epochs, `data/precomputed_martell` dataset) so the leaderboard stays comparable.

| Driver | Architecture | Loss |
|---|---|---|
| `train_martell_mse.py` | `SnowWideEnhancer` (9-ch metadata) | MSE |
| `train_martell_hybrid.py` | `SnowWideEnhancer` (9-ch) | original on Y + chroma_weight·MSE on UV |
| `train_snow_wide_9ch.py` | `SnowWideEnhancer` (9-ch) | original Martell loss (multi-term) |
| `train_stenet.py` | `STENet2024` (RFS + PFE) | MSE + 0.1·MSE on synthesis |
| `train_bi_conv_lstm.py` | `BiConvLSTMEnhancer` (no metadata) | MSE |

## Key findings

1. **The original Martell loss destroys chroma.** The MS-SSIM, Sobel, and Laplacian terms are all luminance-biased (Y has stronger edges). Switching to pure MSE (Martell-MSE) recovers chroma but halves the Y improvement. The hybrid (Y multi-term + chroma MSE, `chroma_weight=1.0`) recovers most of the Y benefit while keeping chroma neutral or slightly positive.

2. **Single-QP training caps chroma improvement.** All architectures (Martell, STENet, Bi-ConvLSTM, VVC-PPFF) plateau at ~−2 % chroma BD-rate on this dataset. The fine-tune experiment (`martell_hybrid_ft`) overfit chroma to the QP=32 distribution and regressed on the unbiased multi-QP test set.

3. **Architecture matters for Y, not chroma.** STENet (761 K) and Bi-ConvLSTM (364 K) both converge to identity Y on single-QP MSE — they don't have enough capacity / inductive bias. Only Martell's `SnowWideEnhancer` (attention fusion + alignment + wide context) extracts a substantial Y gain.

4. **Test set bias is real.** The repo's `dataset_blackfyre` did a sample-level random split (seed 42) across all 53 training videos, so the existing val/test splits are useless. Every BD-rate number above is on truly unseen sequences (not in `utils/fetch_dataset.SELECTED_VIDEOS`).

## Methodology notes

- Bitrate from `.vvc` filesize × 8 / num_frames × source_fps (kbps).
- Y PSNR at native resolution.
- Chroma PSNR uses the same bilinear-up / 2×2 avg-pool round-trip as the model's input pipeline. An identity-model control across all 50 video-QP pairs measured the round-trip artifact at ±0.07 dB max — does not flip any qualitative conclusion.
- Bjontegaard BD-PSNR / BD-Rate use the standard cubic-poly formulation in `bdrate.py` (sanity-tested: identical curves → 0 dB / 0 %; +1 dB shift → −20.6 %).

## Repo state

The branch was created off `lstm-enhancement` and rebuilt from a heavily-polluted state:
- ~150 GB of stale outputs (`output_qp{22-42}/`, `output_test/`, `data_qp{*}_test/`) removed.
- 350+ duplicated/broken eval scripts (`eval_*.py`, `bdrate_*.py`, `calc_*.py`, etc.) removed.
- Single canonical entry points kept: see "Pipeline" and "Training a new model" tables above.

## References

- VVC-PPFF: *Versatile Video Coding-Post Processing Feature Fusion*, Appl. Sci. 2024.
- STENet: *Joint Reference Frame Synthesis and Post Filter Enhancement for VVC*, [arXiv:2404.18058](https://arxiv.org/abs/2404.18058).
- QG-ConvLSTM: Yang et al., *Quality-Gated Convolutional LSTM for Enhancing Compressed Video*, ICME 2019, [arXiv:1903.04596](https://arxiv.org/abs/1903.04596) / [GitHub](https://github.com/ryangchn/QG-ConvLSTM).
