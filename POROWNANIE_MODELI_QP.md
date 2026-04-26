# Porównanie Eksperymentalnych Modeli CNN do Wzmocnienia Wideo VVC

**Data:** 14.04.2026  
**Autor:** Filip  
**Temat pracy:** Poprawa jakości wideo kodowanego VVC przy użyciu sieci CNN

---

## 1. Streszczenie

Dokument przedstawia porównanie eksperymentalnych modeli CNN do poprawy jakości wideo skompresowanego za pomocą standardu VVC (Versatile Video Coding). Przetestowano trzy architektury przy różnych wartościach QP (Quantization Parameter).

---

## 2. Architektury Modeli

### 2.1 ResNet (Intra-only) - Model bazowy

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
       ├── Feature Extraction
       │   └── Conv2D(3→64, 7×7) → BatchNorm → PReLU
       │
       ├── Residual Blocks (×4)
       │   └── Conv2D(64→64, 3×3) → BN → PReLU → Conv2D(64→64, 3×3) → BN → PReLU
       │
       └── Output
           └── Conv2D(64→3, 3×3) → Add(YUV)
```

**Charakterystyka:**
- ✅ Najmniejszy model (414k parametrów)
- ✅ Szybka inferencja
- ❌ Brak wykorzystania informacji temporalnej

---

### 2.2 Snow - Model z fuzją temporalną

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
       │   └── Conv2D(64×2→64) → Conv2D(64→64, 3×3)
       │
       ├── Attention Fusion
       │   └── Conv2D(64×3→64) → Conv2D(64→3) → Sigmoid
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

---

### 2.3 Snow-Wide - Model z rozszerzonym kontekstem

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
- ✅ Najlepsze wyniki PSNR
- ✅ Lepsza rekonstrukcja dużych bloków VVC

---

## 3. Funkcje Strat (Loss Functions)

### 3.1 ResNet (Intra-only)
```
Loss = CharbonnierLoss(YUV)
     = mean(sqrt((enhanced - original)² + ε²))
```

### 3.2 Snow
```
Loss = L1 Loss
     = mean(|enhanced - original|)
```

### 3.3 Snow-Wide (końcowa wersja)
```python
Loss = 0.5 * L1 + 0.15 * MS-SSIM + 0.2 * GradientLoss + 0.15 * LaplacianLoss
```

| Składnik | Waga | Opis |
|----------|------|------|
| L1 Loss | 0.50 | Podstawowa różnica pikseli |
| MS-SSIM | 0.15 | Multi-Scale SSIM dla percepcyjnej jakości |
| Gradient Loss | 0.20 | Sobel filter - preservacja krawędzi |
| Laplacian Loss | 0.15 | Laplacian filter - ostrość detali |

---

## 4. Wyniki Ewaluacji

### 4.1 QP = 22 (niska kompresja, wysoka jakość)

| Model | PSNR Gain | Parametry | Uwagi |
|-------|-----------|-----------|-------|
| ResNet_Intra | +0.13 dB | 414,702 | Słaby wynik |
| **Snow_Wide** | **+0.45 dB** | 1,293,024 | Najlepszy |
| Snow | +0.21 dB | 981,594 | Dobry |

### 4.2 QP = 32 (domyślne)

| Model | PSNR Gain | SSIM | Parametry |
|-------|-----------|------|-----------|
| **ResNet_Intra** | -3.95 dB | 0.9536 | 414,702 |
| Snow | +0.42 dB | 0.9610 | 981,594 |
| **Snow_Wide** | **+0.54 dB** | 0.9632 | 1,293,024 |

### 4.3 Podsumowanie wyników

```
PSNR Gain [dB]
     ^
+0.6 |                                    ████
     |                               ████
+0.4 |                          ████  ████
     |                     ████
+0.2 |                ████
     |           ████
  0.0 |------████--------------------------------> Model
     |    ██
-2.0 | ██
     |
-4.0 |██
     +----------------------------------------+
     ResNet    Snow    Snow_Wide
     
     ████ = Snow_Wide (najlepszy)
     ████ = Snow
     ██ = ResNet
```

---

## 5. Wnioski

### 5.1 Główne wnioski

1. **Ramki temporalne są niezbędne:** Model bezramek F-1/F+1 (ResNet) pogarsza jakość
2. **Wide Context poprawia wyniki:** Dodanie modułu 7×7 z dilation=2 daje +0.12 dB gain vs Snow
3. **Funkcje straty mają znaczenie:** Kombinacja L1 + MS-SSIM + Gradient + Laplacian daje najlepsze wyniki
4. **Snow_Wide osiąga najlepsze wyniki:** +0.45~0.54 dB PSNR gain

### 5.2 Rekomendacje

| Priorytet | Rekomendacja | Oczekiwany zysk |
|-----------|--------------|-----------------|
| Wysoki | Dodać więcej epok treningu (1000+) | +0.1 dB |
| Wysoki | Przetestować na większej liczbie QP | Weryfikacja |
| Średni | DenseNet zamiast ResNet | +0.5 dB (wg Piotra) |

---

## 6. Szczegóły techniczne

### 6.1 Konfiguracja VVC

```yaml
QP: [22, 27, 32, 37, 42]
ALF: 0 (wyłączony)
SAO: 0 (wyłączony)
LoopFilterDisable: 1 (deblocking wyłączony)
Preset: fast
```

### 6.2 Konfiguracja treningu

```yaml
batch_size: 8
epochs: 500
learning_rate: 1e-4
patch_size: 132
optimizer: Adam
scheduler: MultiStepLR (milestones=[50,100,150,200,300])
```

### 6.3 Struktura metadanych (19 kanałów)

| Indeks | Nazwa | Opis |
|--------|-------|------|
| 0 | QP | Quantization Parameter |
| 1-4 | MV_X, MV_Y | Motion Vectors |
| 5-8 | MV_ref | MV referencyjne |
| 9 | Depth | Głębokość CU |
| 10 | PredMode | Tryb predykcji |

---

## 7. Checkpointy

```
checkpoints/
├── snow_epoch_490.pt              (Snow - najlepszy)
├── snow_wide_epoch_460.pt         (Snow-Wide - najlepszy)
└── experiments/enhancer/
    └── vtm_resnet_v6.pth          (ResNet Intra-only)
```

---

*Wygenerowano: 14.04.2026*
