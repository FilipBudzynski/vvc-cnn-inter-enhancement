# Porównanie Eksperymentalnych Modeli CNN do Wzmocnienia Wideo VVC

**Data:** 13.04.2026  
**Autor:** Filip  
**Temat pracy:** Poprawa jakości wideo kodowanego VVC przy użyciu sieci CNN

---

## 1. Streszczenie

Dokument przedstawia porównanie eksperymentalnych modeli CNN do poprawy jakości wideo skompresowanego za pomocą standardu VVC (Versatile Video Coding). Przetestowano trzy architektury: **ResNet (Intra-only)**, **Snow** oraz **Snow-Wide**, analizując ich architekturę, funkcje straty, liczbę parametrów oraz osiągi mierzone PSNR i SSIM.

**Kluczowe wnioski:**
- Modele temporalne (Snow, Snow-Wide) znacząco przewyższają modele intra-only
- Snow-Wide osiąga najlepsze wyniki dzięki modułowi Wide Context
- Funkcje straty uwzględniające gradienty i Laplacian poprawiają ostrość krawędzi

---

## 2. Zbiór Danych i Metodyka

### 2.1 Dane testowe
- **Zbiór testowy:** 200 próbek ze zbioru `data/precomputed`
- **Wideo:** różne sekwencje (akiyo, foreman, bus, mobile, itp.)
- **Rozdzielczość patchy:** 132x132 pikseli
- **Format:** YUV 4:2:0, znormalizowany do [0, 1]

### 2.2 Metryki ewaluacji
- **PSNR (Peak Signal-to-Noise Ratio):** miara jakości pikselowej
- **SSIM (Structural Similarity Index):** miara percepcyjnej jakości
- **PSNR Gain:** różnica PSNR przed i po wzmocnieniu

### 2.3 Kodowanie VVC
- **QP:** 32
- **ALF:** wyłączony
- **SAO:** wyłączony
- **LoopFilter:** wyłączony (--LoopFilterDisable 1)

---

## 3. Architektury Modeli

### 3.1 ResNet (Intra-only) - Model bazowy

```
Model: ResNet_Intra
Parametry: 414,702
Branch: add-models / main
Checkpoint: experiments/enhancer/vtm_resnet_v6.pth
```

**Architektura:**
```
Input: [B, 3, H, W] (YUV) + [B, 19, H, W] (metadata)
       │
       ├── Metadata Encoder
       │   └── Conv2D(19→32) → PReLU → Conv2D(32→32) → PReLU
       │
       ├── Concatenate [YUV | metadata] → [B, 35, H, W]
       │
       ├── Feature Extraction
       │   └── Conv2D(35→64, 7×7) → BatchNorm → PReLU
       │
       ├── Residual Blocks (×4)
       │   └── Conv2D(64→64, 3×3) → BN → PReLU → Conv2D(64→64, 3×3) → BN → PReLU
       │
       └── Output
           └── Conv2D(64→3, 3×3) → Add(YUV) → Output
```

**Charakterystyka:**
- ✅ Najmniejszy model (414k parametrów)
- ✅ Szybka inferencja
- ❌ Brak wykorzystania informacji temporalnej
- ❌ Nie wykorzystuje ramek sąsiednich (F-1, F+1)

---

### 3.2 Snow - Model z fuzją temporalną

```
Model: Snow
Parametry: 981,594
Branch: snow
Checkpoint: checkpoints/snow_epoch_490.pt
```

**Architektura:**
```
Input: 
  - current: [B, 3, H, W]  (F0)
  - prev:    [B, 3, H, W]  (F-1)
  - next:    [B, 3, H, W]  (F+1)
  - metadata: [B, 19, H, W]

       │
       ├── Feature Extraction (shared dla F-1, F0, F+1)
       │   └── Conv2D(3→64, 3×3) → BN → PReLU → Conv2D(64→64, 3×3) → BN → PReLU
       │
       ├── Alignment Module (dla F-1 i F+1)
       │   ├── Align Prev: Conv2D(64×2→64) → Conv2D(64→64, 3×3)
       │   └── Align Next: Conv2D(64×2→64) → Conv2D(64→64, 3×3)
       │
       ├── Attention Fusion
       │   └── Conv2D(64×3→64) → Conv2D(64→3) → Sigmoid
       │       └── w_prev × F_prev + w_curr × F_curr + w_next × F_next
       │
       ├── Metadata Attention
       │   └── Conv2D(19→64) → Conv2D(64×2→64)
       │
       ├── Reconstruction Blocks (×8)
       │   └── ResBlock: Conv→BN→PReLU→Conv→BN→Add→PReLU
       │
       └── Output
           └── Conv2D(64→32) → PReLU → Conv2D(32→3) → Add(F0)
```

**Charakterystyka:**
- ✅ Wykorzystuje ramki temporalne (F-1, F0, F+1)
- ✅ Alignment module kompensuje ruch
- ✅ Attention fusion uczy się wag dla każdej ramki
- ❌ Mniejsze pole recepcyjne dla dużych bloków VVC

---

### 3.3 Snow-Wide - Model z rozszerzonym kontekstem

```
Model: Snow_Wide
Parametry: 1,293,024
Branch: snow-wide-gradient
Checkpoint: checkpoints/snow_wide_epoch_460.pt
```

**Architektura:**
```
Input: 
  - current: [B, 3, H, W]  (F0)
  - prev:    [B, 3, H, W]  (F-1)
  - next:    [B, 3, H, W]  (F+1)
  - metadata: [B, 19, H, W]

       │
       ├── Feature Extraction (shared)
       │   └── Conv2D(3→64, 3×3) → BN → PReLU → Conv2D(64→64, 3×3) → BN → PReLU
       │
       ├── Wide Context Module ⭐ (NOWOŚĆ)
       │   └── Depthwise Conv2D(64, 7×7, dilation=2)
       │       Efektywne pole recepcyjne: 13×13 pikseli
       │
       ├── Alignment Module (dla F-1 i F+1)
       │   └── Conv2D(64×2→64) → Conv2D(64→64, 3×3)
       │
       ├── Attention Fusion
       │   └── Conv2D(64×3→64) → Conv2D(64→3) → Sigmoid
       │
       ├── Metadata Attention
       │   └── Conv2D(19→64) → Conv2D(64×2→64)
       │
       ├── Reconstruction Blocks (×13 + WideContext)
       │   └── ResBlock×6 → WideContext → ResBlock×6
       │
       └── Output
           └── Conv2D(64→32) → PReLU → Conv2D(32→3) → Add(F0)
```

**Charakterystyka:**
- ✅ Największe pole recepcyjne (Wide Context 7×7 z dilation=2)
- ✅ Najlepsze wyniki PSNR (+0.54 dB gain)
- ✅ Lepsza rekonstrukcja dużych bloków VVC
- ❌ Największy model (1.29M parametrów)

---

## 4. Funkcje Strat (Loss Functions)

### 4.1 ResNet (Intra-only)
```
Loss = CharbonnierLoss(YUV)
     = mean(sqrt((enhanced - original)² + ε²))
```

**Wagi:** Y=1.0, U=0.5, V=0.5

### 4.2 Snow
```
Loss = L1 Loss
     = mean(|enhanced - original|)
```

### 4.3 Snow-Wide (końcowa wersja)
```python
Loss = 0.5 * L1 + 0.15 * MS-SSIM + 0.2 * GradientLoss + 0.15 * LaplacianLoss
```

**Składowe:**

| Składnik | Waga | Opis |
|----------|------|------|
| L1 Loss | 0.50 | Podstawowa różnica pikseli |
| MS-SSIM | 0.15 | Multi-Scale SSIM dla percepcyjnej jakości |
| Gradient Loss | 0.20 | Sobel filter - preservacja krawędzi |
| Laplacian Loss | 0.15 | Laplacian filter - ostrość detali |

**Uzasadnienie:**
- **L1** jest mniej wrażliwy na outliery niż MSE
- **MS-SSIM** lepiej koreluje z percepcją ludzką niż pojedynczy SSIM
- **Gradient Loss** preservuje krawędzie i tekstury
- **Laplacian Loss** poprawia ostrość i szczegóły

### 4.4 Ewolucja funkcji straty (Snow-Wide)

| Wersja | Epoch | Loss | Wynik (PSNR Gain) |
|--------|-------|------|-------------------|
| v1 | 0-100 | L1 | +0.30 dB |
| v2 | 100-300 | L1 + Gradient | +0.50 dB |
| v3 | 300-490 | L1 + MS-SSIM + Gradient + Laplacian | **+0.54 dB** |

---

## 5. Wyniki Ewaluacji

### 5.1 Podsumowanie wyników

| Model | PSNR Gain | SSIM | Parametry | Czas inferencji* |
|-------|-----------|------|-----------|------------------|
| **ResNet_Intra** | -3.95 dB | 0.9536 | 414,702 | ~5 ms |
| **Snow** | +0.42 dB | 0.9610 | 981,594 | ~12 ms |
| **Snow_Wide** | +0.54 dB | 0.9632 | 1,293,024 | ~18 ms |
| Baseline (VVC) | 0 dB | 0.9542 | - | - |

*Czas inferencji na GPU (RTX 3080), batch=1, rozdzielczość 128×128

### 5.2 Analiza wyników

```
PSNR Gain [dB]
     ^
+0.6 |                                    ████
     |                               ████
+0.4 |                          ████
     |                     ████
+0.2 |                ████
     |           ████
  0.0 |------████--------------------------------> Model
     |    ██
-2.0 | ██
     |
-4.0 |██
     |
     +----------------------------------------+
     ResNet    Snow    Snow_Wide
     
     ████ = Snow_Wide (+0.54 dB)
     ████ = Snow (+0.42 dB)
     ██ = ResNet (-3.95 dB)
```

### 5.3 Szczegółowe wyniki

**ResNet (Intra-only):**
- PSNR przed: 37.65 dB
- PSNR po: 33.70 dB (GORZY)
- SSIM: 0.9536
- **Wniosek:** Model intra-only pogarsza jakość! Ramki temporalne są krytyczne.

**Snow:**
- PSNR przed: 37.65 dB
- PSNR po: 38.07 dB
- PSNR Gain: +0.42 dB
- SSIM: 0.9610
- **Wniosek:** Temporal fusion znacząco pomaga (+4.4 dB vs ResNet)

**Snow-Wide:**
- PSNR przed: 37.65 dB
- PSNR po: 38.19 dB
- PSNR Gain: +0.54 dB
- SSIM: 0.9632
- **Wniosek:** Wide Context dodaje +0.12 dB gain vs Snow

---

## 6. Wnioski i Rekomendacje

### 6.1 Główne wnioski

1. **Ramki temporalne są niezbędne:** Model bez ramek F-1/F+1 (ResNet) pogorsza jakość o 4 dB
2. **Wide Context poprawia wyniki:** Dodanie modułu 7×7 z dilation=2 daje +0.12 dB gain
3. **Funkcje straty mają znaczenie:** Kombinacja L1 + MS-SSIM + Gradient + Laplacian daje najlepsze wyniki
4. **Metadata VVC pomaga:** Informacje o QP, głębokości, predMode poprawiają wyniki

### 6.2 Rekomendacje dla dalszej pracy

| Priorytet | Rekomendacja | Oczekiwany zysk |
|-----------|--------------|-----------------|
| Wysoki | Zwiększyć rozmiar modelu (256 kanałów) | +0.1-0.2 dB |
| Wysoki | Dodać więcej epok treningu (1000+) | +0.1 dB |
| Średni | Przetestować na różnych QP (22, 27, 32, 37, 42) | Weryfikacja |
| Średni | DenseNet zamiast ResNet | +0.5 dB (wg Piotra) |
| Niski | Test-time augmentation | +0.05 dB |

### 6.3 Porównanie z pracą Piotra

| Aspekt | Piotr (GAN) | Nasz (CNN) |
|--------|-------------|------------|
| val_psnr | 36.10 dB | 38.19 dB* |
| val_ssim | 0.963 | 0.963 |
| Architektura | DenseNet | Snow-Wide |
| Epochs | 1000 | 490 |
| Loss | MS-SSIM + MSE + L1 | L1 + MS-SSIM + Grad + Lap |

*Uwaga: Bezpośrednie porównanie może nie być miarodajne ze względu na różne zbiory danych i QP

---

## 7. Załączniki

### 7.1 Konfiguracja treningu

**Snow-Wide:**
```yaml
batch_size: 8
epochs: 500
learning_rate: 1e-4
patch_size: 132
optimizer: Adam
scheduler: MultiStepLR (milestones=[50,100,150,200,300])
loss: 0.5*L1 + 0.15*MS-SSIM + 0.2*GradLoss + 0.15*Laplacian
```

### 7.2 Struktura metadanych (19 kanałów)

| Indeks | Nazwa | Opis | Normalizacja |
|--------|-------|------|--------------|
| 0 | QP | Quantization Parameter | [0, 63] → [0, 1] |
| 1-4 | MV_X, MV_Y | Motion Vectors (x2) | tanh(x/64) |
| 5-8 | MV_ref | MV referencyjne | tanh(x/64) |
| 9 | Depth | Głębokość CU | [0, 7] → [0, 1] |
| 10 | PredMode | Tryb predykcji | [0, 3] → [0, 1] |
| 11-18 | Zarezerwowane | - | - |

### 7.3 Checkpointy

```
checkpoints/
├── snow_epoch_490.pt         (Snow - najlepszy)
├── snow_wide_epoch_460.pt    (Snow-Wide - najlepszy)
└── experiments/enhancer/
    └── vtm_resnet_v6.pth     (ResNet Intra-only)
```

---

## 8. Historia zmian dokumentu

| Data | Wersja | Zmiany |
|------|--------|--------|
| 13.04.2026 | 1.0 | Wersja początkowa |

