# Deepfake Video Detection — Experiment Report

Spatial-temporal CNN/Transformer forensics on FaceForensics++ (5 generators) and Celeb-DF v2 (held-out cross-dataset). CPU training (16-core, no GPU).

> **Protocol integrity:** all numbers below use video-level-disjoint splits (the Kaggle mirror's shipped splits leaked 124 val + 78 test videos into train; 1,044 leaked rows were removed) and the temporal model is length-controlled (T=2 frames per clip — the raw data has a frame-count shortcut: real clips carry 12 frames, fakes 2–3, which alone yields a fake-perfect AUC of 1.0).

## Headline results

| Model | Val AUC | Notes |
|---|---|---|
| Spatial (EfficientNet-B0, frame-level) | 0.9100 | epoch 2 |
| Video-level: spatial_agg_mean | 0.9553 | |
| Video-level: spatial_agg_max | 0.9336 | |
| Video-level: spatial_agg_topk | 0.9336 | |
| Video-level: temporal_transformer | 0.9456 | |

## Cross-generator generalization (frame AUC)

Research question: *which artifacts generalize across generators?*

### Cross-source held-out test (Celeb-DF v2 vs FF++)

| Generator / source | AUC | Acc | n |
|---|---|---|---|
| CelebDF | 0.9220 | 0.8383 | 3000 |
| FF++ | 0.8928 | 0.8810 | 1916 |
| **OVERALL** | **0.9033** | 0.8296 | 3416 |

### Per-generator FF++ test (cross-preprocessing)

| Generator / source | AUC | Acc | n |
|---|---|---|---|
| Deepfakes | 0.8011 | 0.7102 | 1394 |
| Face2Face | 0.6291 | 0.5575 | 1399 |
| FaceShifter | 0.5955 | 0.5536 | 1398 |
| FaceSwap | 0.5629 | 0.5186 | 1398 |
| NeuralTextures | 0.6220 | 0.5397 | 1399 |
| **OVERALL** | **0.6419** | 0.3578 | 4192 |

## Artifact analysis

### Spectral fingerprints (radial FFT divergence vs real)

Higher L1 = stronger deviation from real. Band columns show WHERE in the frequency spectrum the artifact lives.

| Generator | L1 | low-freq | mid-freq | high-freq |
|---|---|---|---|---|
| FF++ | 0.0543 | 0.0226 | 0.0070 | 0.0247 |
| CelebDF | 0.0239 | 0.0117 | 0.0034 | 0.0088 |

### Detector attention (Grad-CAM center mass)

Fraction of attention mass in the central face region. Similar values across generators ⇒ the model uses a consistent, generalizable cue.

| Generator | center mass | n |
|---|---|---|
| CelebDF | 0.321 ± 0.157 | 120 |
| FF++ | 0.223 ± 0.109 | 120 |
| real | 0.094 ± 0.133 | 120 |

## Spatial training history

| epoch | train loss | val AUC | val acc |
|---|---|---|---|
| 1 | 0.5348 | 0.8750 | 0.7991 |
| 2 | 0.4450 | 0.9100 | 0.8342 |
