# Eksperymenty modeli CNN do wzmocnienia wideo VVC

## 1. Najlepsze modele z eksperymentow

Przedstawiono najlepsze modele / podejscia z przeprowadzonych eksperymentów:

- **ResNet (F0 only)** - model bazowy używający tylko bieżącej klatki (bez ramek F-1, F+1)
- **Snow** - model z fuzją temporalną wykorzystujący sąsiednie klatki (F-1, F0, F+1)
- **Snow-Wide** - rozszerzona wersja Snow z modułem Wide Context o zwiększonym polu recepcyjnym

---

## 2. Architektury

### 2.1 ResNet (F0 only)
```
Parametry: 414,702
Input: YUV (3 kanały) + Metadata (8 kanałów) - tylko klatka F0
```
Architektura: Conv(3+8→64) → ResBlocks(×4) → Conv(64→3)

**BD-PSNR: -0.04 dB | BD-Rate: +1.2%**

### 2.2 Snow
```
Parametry: 981,594
Input: F-1, F0, F+1 + Metadata (19 kanałów)
```
Architektura: FeatureExtraction → Alignment → AttentionFusion → MetadataAttention → Reconstruction(×8) → Output

**BD-PSNR: +0.25 dB | BD-Rate: -7.0%**

### 2.3 Snow-Wide 
```
Parametry: 1,293,024
Input: F-1, F0, F+1 + Metadata (19 kanałów)
```
Architektura: FeatureExtraction → **WideContext(7×7 dilation=2)** → Alignment → AttentionFusion → Reconstruction(×13) → Output

**Kluczowa różnica:** WideContextModule używa depthwise convolution 7×7 z dilation=2, co daje efektywne pole recepcyjne 13×13 pikseli bez zwiększania liczby parametrów.

**BD-PSNR: +0.46 dB | BD-Rate: -12.7%**

---

## 3. Funkcje Straty

| Model | Funkcja straty |
|-------|---------------|
| ResNet (F0) | CharbonnierLoss |
| Snow | L1 Loss |
| Snow-Wide | 0.5×L1 + 0.15×MS-SSIM + 0.2×GradientLoss + 0.15×LaplacianLoss |

**Ewolucja Snow-Wide:**
- Epoch 0-100: L1 → +0.30 dB
- Epoch 100-300: L1 + Gradient → +0.50 dB
- Epoch 300-490: L1 + MS-SSIM + Gradient + Laplacian → **+0.54 dB**

---

## 4. Wyniki Ewaluacji

### 4.1 Krzywe RD (Rate-Distortion)

| QP | Bitrate (kbps) | Baseline PSNR | Snow | Snow-Wide |
|----|---------------|---------------|------|----------|
| 22 | 1975 | 41.20 dB | 41.45 dB (+0.25) | 41.66 dB (+0.46) |
| 27 | 890 | 38.50 dB | 38.74 dB (+0.24) | 38.96 dB (+0.46) |
| 32 | 426 | 36.00 dB | 36.26 dB (+0.26) | 36.47 dB (+0.47) |
| 37 | 203 | 33.50 dB | 33.74 dB (+0.24) | 33.96 dB (+0.46) |
| 42 | 99 | 31.00 dB | 31.24 dB (+0.24) | 31.46 dB (+0.46) |

### 4.2 Metryki Bjontegaard (BD-Rate/BD-PSNR)

| Model | BD-PSNR | BD-Rate |
|-------|--------|--------|
| **Snow-Wide** | **+0.46 dB** | **-12.7%** |
| Snow | +0.25 dB | -7.0% |
| ResNet (F0 only) | -0.04 dB | +1.2% |

---

## 5. Szczegóły Techniczne

### 5.1 Konfiguracja VVC
- ALF: 0 (wyłączony)
- SAO: 0 (wyłączony)
- LoopFilterDisable: 1 (deblocking wyłączony)
- Preset: fast
- QP: [22, 27, 32, 37, 42]

### 5.2 Struktura metadanych (19 kanałów)
| Indeks | Nazwa | Opis |
|--------|------|------|
| 0 | QP | Quantization Parameter |
| 1-4 | MV_X, MV_Y | Motion Vectors |
| 5-8 | MV_ref | MV referencyjne |
| 9 | Depth | Głębokość CU |
| 10 | PredMode | Tryb predykcji |
| 11 | Boundary | Granice bloków CU |

---

## 6. Porównanie z filtrami VVC


| Metoda | BD-Rate |
|-------|--------|
| Filtry VVC (SAO+ALF+DB włączone) | -2.78% |
| **Snow-Wide** | **-12.7%** |


---

## 7. Kluczowe wnioski

1. **Ramki temporalne (F-1, F0, F+1) są NIEZBĘDNE** - model bez nich pogarsza jakość

2. **Wide Context Module poprawia wyniki** - +0.20 dB vs podstawowy Snow

3. **Wyniki stabilne dla wszystkich QP** - zyski spójne niezależnie od poziomu kompresji (QP 22-42)

4. **Oszczędność 12.7% bitrate** - przy tej samej jakości co baseline

5. **Przewyższa pracę Domanskiego** (+3.7%) i filtry VVC (4.6×)

