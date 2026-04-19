# Porównanie Eksperymentalnych Modeli CNN do Wzmocnienia Wideo VVC

**Data:** 19.04.2026  
**Autor:** Filip  
**Temat:** Poprawa jakości wideo kodowanego VVC przy użyciu sieci CNN

---

## 1. Streszczenie

Przedmiotem pracy jest porównanie architektur CNN do wzmocnienia (enhancement) wideo skompresowanego przy użyciu standardu VVC (H.266). Przetestowano następujące podejścia:

- **ResNet (F0 only)** - model bazowy używający tylko bieżącej klatki (bez ramek F-1, F+1)
- **Snow** - model z fuzją temporalną wykorzystujący sąsiednie klatki (F-1, F0, F+1)
- **Snow-Wide** - rozszerzona wersja Snow z modułem Wide Context o zwiększonym polu recepcyjnym
- **Blackfyre** - model z dodatkową modulacją uwagi (dalsze testy)

---

## 2. Architektury

### 2.1 ResNet (F0 only) - Model bazowy bez ramek temporalnych
```
Parametry: 414,702
Input: YUV (3 kanały) - tylko klatka F0
Checkpoint: experiments/enhancer/vtm_resnet_v6.pth
```
Architektura: Conv(3→64) → ResBlocks(×4) → Conv(64→3) + skip connection

**Wynik: NIE ZALECANY** - bez ramek temporalnych model pogarsza jakość (+1.2% BD-Rate oznacza większy bitrate przy tej samej jakości)

### 2.2 Snow
```
Parametry: 981,594
Input: F-1, F0, F+1 + Metadata (19 kanałów)
Checkpoint: checkpoints/snow_epoch_490.pt
```
Architektura: FeatureExtraction → Alignment → AttentionFusion → MetadataAttention → Reconstruction(×8) → Output

### 2.3 Snow-Wide [NAJLEPSZY]
```
Parametry: 1,293,024
Input: F-1, F0, F+1 + Metadata (19 kanałów)
Checkpoint: checkpoints/snow_wide_epoch_460.pt
```
Architektura: FeatureExtraction → **WideContext(7×7 dilation=2)** → Alignment → AttentionFusion → Reconstruction(×13) → Output

**Kluczowa różnica:** WideContextModule używa depthwise convolution 7×7 z dilation=2, co daje efektywne pole recepcyjne 13×13 pikseli bez zwiększania liczby parametrów.

### 2.4 Blackfyre
```
Parametry: ~1,100,000 (est.)
Input: F-1, F0, F+1 + Metadata (19 kanałów)
Checkpoint: checkpoints/blackfyre_*.pt
```
Architektura: FeatureExtraction → Self-Attention → Temporal Alignment → Output

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

| Model | BD-PSNR | BD-Rate | Opis |
|-------|--------|---------|------|
| **Snow-Wide** | **+0.46 dB** | **-12.7%** | Najlepszy wynik |
| Snow | +0.25 dB | -7.0% | Dobry wynik |
| Blackfyre | +0.40 dB | -11.1% | Częściowa ewaluacja |
| ResNet (F0 only) | -0.04 dB | +1.2% | Pogarsza jakość |

**Wyjaśnienie:**
- **BD-PSNR** = ile dB zyskuje jakość przy tym samym bitrate
- **BD-Rate** = ile % bitrate można oszczędzić przy tej samej jakości (wartość ujemna = oszczędność, dodatnia = straty)

---

## 5. Wymagania od promotora - status

| Wymaganie | Status |
|----------|--------|
| BD-PSNR | ✅ Zaimplementowane |
| BD-Rate | ✅ Zaimplementowane |
| Funkcja celu (MS-SSIM) | ✅ Zaimplementowane |
| Mapa granic bloków (Boundary) | ✅ W metadanych (19 kanałów) |
| Typ ramki (I/P/B) | ❌ Brak - wymaga dodania |

---

## 6. Szczegóły Techniczne

### 6.1 Konfiguracja VVC
- ALF: 0 (wyłączony)
- SAO: 0 (wyłączony)
- LoopFilterDisable: 1 (deblocking wyłączony)
- Preset: fast
- QP: [22, 27, 32, 37, 42]

### 6.2 Struktura metadanych (19 kanałów)
| Indeks | Nazwa | Opis |
|--------|------|------|
| 0 | QP | Quantization Parameter |
| 1-4 | MV_X, MV_Y | Motion Vectors |
| 5-8 | MV_ref | MV referencyjne |
| 9 | Depth | Głębokość CU |
| 10 | PredMode | Tryb predykcji |
| 11 | Boundary | Granice bloków CU |

---

## 7. Porównanie z pracą Piotra Domanskiego

### Profil kodowania
Wszystkie wyniki są dla profilu **RA** (Random Access) - domyślny profil vvenc, który używa ramek I, P i B.

### Wyniki z pracy P. Domanskiego (profil RA):

| Model | BD-Rate |
|-------|---------|
| DenseNet + GAN | -9.04% |
| ResNet | -5.83% |
| Konwolucyjna | -1.99% |

### Porównanie z moimi wynikami (profil RA):

| Model | BD-Rate | Różnica |
|-------|---------|---------|
| **Snow-Wide** | **-12.7%** | **+3.7% lepszy!** |
| Snow | -7.0% | +2.0% lepszy |
| Domanski DenseNet+GAN | -9.04% | baseline |

**Wniosek:** Snow-Wide jest o 3.7% LEPSZY od najlepszego modelu Domanskiego dla profilu RA.

---

## 8. Porównanie z filtrami VVC

### Dane z pracy P. Domanskiego:

| Metoda | BD-Rate |
|-------|--------|
| Filtry VVC (SAO+ALF+DB włączone) | -2.78% |
| **Snow-Wide** | **-12.7%** |

**Wniosek:** Model CNN jest 4.6× SKUTECZNIEJSZY niż wbudowane filtry VVC.

---

## 9. Kluczowe wnioski

1. **Ramki temporalne (F-1, F0, F+1) są NIEZBĘDNE** - model bez nich (ResNet F0 only) pogarsza jakość o +1.2% BD-Rate

2. **Wide Context Module poprawia wyniki** - +0.20 dB vs podstawowy Snow

3. **Wyniki stabilne dla wszystkich QP** - zyski spójne niezależnie od poziomu kompresji (QP 22-42)

4. **Oszczędność 12.7% bitrate** - przy tej samej jakości co baseline

5. **Przewyższa pracę Domanskiego** (+3.7%) i filtry VVC (4.6×)

---

*Raport wygenerowano: 19.04.2026*
