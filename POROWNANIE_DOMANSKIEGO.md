# Porównanie z pracą Piotra Domanskiego (do wiadomości do promotora)

## Profil kodowania
Wszystkie wyniki są dla profilu RA (Random Access) - domyślny profil vvenc, który używa ramek I, P i B.

## Wyniki z pracy P. Domanskiego (profil RA):
| Model | BD-Rate |
|-------|---------|
| DenseNet + GAN | -9.04% |
| ResNet | -5.83% |
| Konwolucyjna | -1.99% |

## Porównanie z moimi wynikami (profil RA):
| Model | BD-Rate | Różnica |
|-------|---------|---------|
| Snow-Wide | -12.7% | +3.7% lepszy! |
| Snow | -7.0% | +2.0% lepszy |
| Domanski DenseNet+GAN | -9.04% | baseline |

## Porównanie z filtrami VVC:
| Metoda | BD-Rate |
|-------|--------|
| Filtry VVC (SAO+ALF+DB włączone) | -2.78% |
| Snow-Wide | -12.7% |

Wniosek: Model CNN jest 4.6× skuteczniejszy niż wbudowane filtry VVC.
