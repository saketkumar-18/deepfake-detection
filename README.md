# 🕵️ Deepfake Video Detection — Spatial-Temporal CNN/Transformer

**Live demo (100% on-device, nothing uploaded):** https://deepfake-detector-amber.vercel.app

**Production-ready deepfake video forensics**: an EfficientNet-B0 spatial detector
scores individual frames, and a temporal transformer aggregates frame embeddings
into a video-level verdict. Includes a cross-generator generalization study and
artifact analysis (FFT spectral fingerprints + Grad-CAM attention) answering the
research question: *which manipulation artifacts generalize across generators?*

## Results

All numbers on a **video-level-disjoint held-out test split** (no identity/video
leakage), CPU-only training.

| Model / aggregation | Test Video AUC | AP | Acc |
|---|---|---|---|
| Spatial (EfficientNet-B0), mean-pool | **0.9553** | 0.9838 | 0.9170 |
| Spatial, max-pool | 0.9336 | 0.9764 | 0.8960 |
| Spatial, top-k | 0.9336 | 0.9764 | 0.8960 |
| **Spatial + Temporal Transformer** (T=2, length-controlled) | **0.9456** | 0.9768 | 0.8813 |

Spatial detector alone: val AUC **0.9100** (head-only 0.7910 → fine-tune 0.8750 → 0.9100).
Temporal transformer: val AUC **0.9367** under the length-control protocol.

**Cross-source generalization** (one model, held-out test, per source):

| Source | Test AUC | n |
|---|---|---|
| Celeb-DF v2 (out-of-distribution generator family) | 0.9220 | 3000 |
| FaceForensics++ | 0.8928 | 1916 |
| **Overall** | **0.9033** | 5486 |

**Cross-preprocessing generalization** (FF++ frames from a *different* face-crop
pipeline than training — the hard, realistic setting):

| Generator | Test AUC | n |
|---|---|---|
| Deepfakes | 0.8011 | 1394 |
| Face2Face | 0.6291 | 1399 |
| NeuralTextures | 0.6220 | 1399 |
| FaceShifter | 0.5955 | 1398 |
| FaceSwap | 0.5629 | 1398 |
| **Overall** | **0.6419** | 6988 |

> **Key finding:** artifacts generalize well *within* a preprocessing pipeline
> (AUC ≈ 0.90–0.96) but degrade sharply *across* pipelines/generators
> (AUC ≈ 0.56–0.80). Swap-based forgeries (FaceSwap) leave the weakest
> spectral/spatial cue; autoencoder-based (Deepfakes) the strongest.

### Which artifacts generalize (FFT + Grad-CAM)

- **Spectral divergence vs. real** (radial power spectrum, L1): FF++ fakes
  deviate **2.3× more** than Celeb-DF fakes overall (L1 0.054 vs 0.024), driven
  by the **high-frequency band** (0.025 vs 0.009) — the classic GAN/decoder
  upsampling fingerprint. Mid-band is the most stable cue across sources.
- **Grad-CAM attention** (center-of-mass, 0=center): fakes pull attention to the
  face interior (Celeb-DF 0.32, FF++ 0.22) while real frames scatter (0.09) —
  the detector keys on manipulated facial content, not borders/compression.

### ⚠️ Two integrity issues found & fixed (documented for reproducibility)

1. **Split leakage in the Kaggle mirror.** The mirror's provided train/val/test
   CSVs leaked **124 val + 78 test videos into train**. We enforce video-level
   disjoint splits in `prepare_data.py` (dropped 1,044 leaked train rows).
2. **Frame-count shortcut.** In this mirror, real clips have 12 frames but fakes
   only 2–3, so a temporal model can score a fake-perfect 1.0 AUC by *counting
   frames*. We add a **length-control protocol** (`--max-frames T`) that
   subsamples every clip to exactly T frames and drops shorter ones, and a
   runtime warning when the shortcut is detected. All temporal numbers above use
   T=2.

## Architecture

```
video ──► 16 evenly-sampled frames ──► face crop (Haar + margin)
                                          │
                    ┌─────────────────────┴──────────────────────┐
                    ▼                                            ▼
        EfficientNet-B0 (ImageNet init)              frame embeddings (T, 1280)
        frame logit → sigmoid → frame score                    │
                                                               ▼
                                          Temporal Transformer (3 layers, 4 heads)
                                          proj→256 + pos-embed + CLS + attention pool
                                                               │
                                                               ▼
                                                   video fake probability
```

## Why these artifacts generalize (research angle)

1. **Spectral fingerprints** — GAN/diffusion decoders (transposed convolutions,
   upsampling grids) leave periodic structure in the mid/high-frequency band of
   the radial power spectrum. We measure per-generator spectral divergence from
   real (`analyze_artifacts.py`) and show which bands deviate consistently.
2. **Spatial attention** — Grad-CAM shows whether the detector attends to the
   central face region (manipulated content) vs. boundaries/compression
   shortcuts, per generator. Consistent placement ⇒ generalizable cue.
3. **Cross-generator matrix** — per-generator AUC of one model trained on the
   FF++ mix, tested on each generator + Celeb-DF v2 (out-of-distribution).

## Setup

```bash
uv venv .venv
# CPU wheels:
VIRTUAL_ENV=.venv uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
VIRTUAL_ENV=.venv uv pip install -e ".[app,dev]"
```

## Data

Primary source: pre-extracted face frames from **FaceForensics++ (C23)** —
5 generators (Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter/DFD) —
plus **Celeb-DF v2** frames as the held-out cross-dataset test.

```bash
# Option A (used here): Kaggle pre-extracted frames
kaggle datasets download wish096/ff-andcelebdf-frame-dataset-by-wish -p data/raw_frames --unzip

# Option B: raw videos → run face-crop preprocessing
python -m deepfake_detection.preprocess --config configs/preprocess.yaml
```

Expected layout under `data/processed/` (auto-detected):

```
real/ or Celeb-real/ or Original/     → label 0
fake/ or Celeb-synthesis/             → label 1 (generator=CelebDF)
Deepfakes/ Face2Face/ FaceSwap/ ...   → label 1 (generator=<dir name>)
```

## Train

```bash
# 0. build leak-free, video-disjoint splits from the Kaggle mirror
python -m deepfake_detection.prepare_data

# 1. spatial detector (frame-level)
python -m deepfake_detection.train_spatial --config configs/spatial.yaml
# fine-tune from the head-only checkpoint on a balanced subset (CPU-friendly)
python -m deepfake_detection.train_spatial --config configs/spatial.yaml \
    --resume checkpoints/spatial_effb0.pt --finetune-only --epochs 2 --balance 5000

# 2. temporal: cache embeddings, then train transformer with length control
python -m deepfake_detection.train_temporal embed --config configs/temporal.yaml
python -m deepfake_detection.train_temporal train --config configs/temporal.yaml --max-frames 2

# 3. benchmark + research analyses
python -m deepfake_detection.benchmark --split test --max-frames 2
python -m deepfake_detection.analyze_generalization --config configs/spatial.yaml \
    --ckpt checkpoints/spatial_effb0.pt --data-root data/processed --split test
python -m deepfake_detection.analyze_artifacts --data-root data/processed/test \
    --ckpt checkpoints/spatial_effb0.pt
```

## Inference

```bash
# CLI
python -m deepfake_detection.inference path/to/video.mp4

# Gradio web demo (http://127.0.0.1:7860)
python -m deepfake_detection.app

# REST API
uvicorn deepfake_detection.api:app --port 8000
# POST /predict with multipart video file
```

### In-browser demo (static, on-device WASM)

The `app/` folder is a zero-backend demo: MediaPipe BlazeFace finds faces and
the EfficientNet-B0 detector runs as **fp16 ONNX in WebAssembly** (onnxruntime-web),
so no video ever leaves the user's device. Deployed to Vercel as static files.

```bash
python -m deepfake_detection.export_onnx   # -> app/assets/model.fp16.onnx (8.1 MB)
# parity verified: |ORT-web WASM - torch| max 4.85e-3, ~45 ms/frame
```

> int8 dynamic quantization was tested and **rejected**: it collapses
> EfficientNet's SE blocks (frame AUC 0.97 → 0.55). fp16 halves the size with
> no measurable accuracy loss.

## Tests

```bash
python -m pytest tests/ -q
```

## Project layout

```
configs/            YAML configs (preprocess, spatial, temporal)
src/deepfake_detection/
  dataset.py        frame/video datasets, tree scanning, augmentation
  models/spatial.py EfficientNet-B0 frame detector
  models/temporal.py temporal transformer over embeddings
  preprocess.py     video → face crops (OpenCV Haar)
  train_spatial.py  frame-level training + AUC eval
  train_temporal.py embedding cache + transformer training
  analyze_generalization.py  per-generator AUC study
  analyze_artifacts.py       FFT spectra + Grad-CAM zones
  inference.py      video → verdict facade
  app.py / api.py   Gradio demo / FastAPI service
  metrics.py        AUC/AP/acc, per-generator, video aggregation
tests/              unit tests (parsing, models, metrics, preprocess)
reports/            JSON + PNG analysis outputs
checkpoints/        model weights
```

## Datasets & citation

- FaceForensics++ (Rössler 2019), Celeb-DF v2 (Li 2020, CVPR).
- Research use only; CC BY-NC 4.0 (Celeb-DF). Detection scores are forensic
  evidence cues, not legal proof.

## License

MIT (code). Model weights inherit dataset licenses for research use.
