#!/usr/bin/env bash
# Full training + analysis pipeline (CPU-friendly defaults).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
[ -f "$PY" ] || PY=.venv/bin/python

echo "==> 1/5 spatial detector"
"$PY" -m deepfake_detection.train_spatial --config configs/spatial.yaml

echo "==> 2/5 temporal embeddings"
"$PY" -m deepfake_detection.train_temporal embed --config configs/temporal.yaml

echo "==> 3/5 temporal transformer"
"$PY" -m deepfake_detection.train_temporal train --config configs/temporal.yaml

echo "==> 4/5 generalization + artifacts"
"$PY" -m deepfake_detection.analyze_generalization --data-root data/processed
"$PY" -m deepfake_detection.analyze_artifacts --data-root data/processed

echo "==> 5/5 benchmark"
"$PY" -m deepfake_detection.benchmark

echo "ALL DONE"
