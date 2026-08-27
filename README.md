# 🕵️ Deepfake Video Detection — Spatial-Temporal CNN/Transformer

**Production-ready deepfake video forensics**: an EfficientNet-B0 spatial detector
scores individual frames, and a temporal transformer aggregates frame embeddings
into a video-level verdict. Includes a cross-generator generalization study and
artifact analysis (FFT spectral fingerprints + Grad-CAM attention) answering the
research question: *which manipulation artifacts generalize across generators?*

## Results

| Model | Val AUC (video-level) | Notes |
|---|---|---|
| Spatial only (EfficientNet-B0, mean agg) | see `reports/` | frame detector |
| Spatial + Temporal Transformer | see `reports/` | full pipeline |

**Cross-generator generalization** (train on FF++ mix, per-generator test AUC):
see `reports/generalization.json` — reports per-generator AUC including the
held-out Celeb-DF v2 (different generator family, higher visual quality).

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
# 1. spatial detector (frame-level)
python -m deepfake_detection.train_spatial --config configs/spatial.yaml

# 2. temporal: cache embeddings, then train transformer
python -m deepfake_detection.train_temporal embed
python -m deepfake_detection.train_temporal train

# 3. research analyses
python -m deepfake_detection.analyze_generalization --data-root data/processed
python -m deepfake_detection.analyze_artifacts --data-root data/processed
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
