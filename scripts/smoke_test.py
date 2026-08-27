"""Synthetic end-to-end smoke test.

Creates a tiny synthetic frame tree (real vs fake with a learnable color
difference), then runs: spatial training (1 epoch, tiny) -> embedding cache ->
temporal training (2 epochs) -> generalization eval -> artifact analysis.
Verifies the whole pipeline wires together on CPU without real data.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SYN = ROOT / "data" / "synthetic"


def make_synth(n_videos_per_class=6, frames_per_video=8, size=64):
    if SYN.exists():
        shutil.rmtree(SYN)
    rng = np.random.default_rng(0)
    for label, cls in [(0, "real"), (1, "Deepfakes"), (1, "Face2Face")]:
        for v in range(n_videos_per_class):
            vid = f"{cls}_v{v}"
            d = SYN / cls
            d.mkdir(parents=True, exist_ok=True)
            base = rng.integers(60, 190, size=3)
            if label == 1:
                base = np.clip(base + 40, 0, 255)  # learnable shift
            for f in range(frames_per_video):
                img = np.clip(base + rng.normal(0, 8, (size, size, 3)), 0, 255).astype(np.uint8)
                Image.fromarray(img).save(d / f"{vid}_f{f}.jpg")
    print(f"[synthetic] created {SYN}")


def run(cmd: list[str]):
    print(f"\n$ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(r.stdout[-2000:])
    if r.returncode != 0:
        print(r.stderr[-2000:])
        raise SystemExit(f"FAILED: {' '.join(cmd)}")


def main():
    make_synth()
    py = sys.executable

    # spatial: 1 epoch on synthetic
    run([py, "-m", "deepfake_detection.train_spatial",
         "--config", "configs/spatial.yaml",
         "--data-root", str(SYN), "--epochs", "1", "--limit", "200"])

    # temporal embed + train
    run([py, "-m", "deepfake_detection.train_temporal", "embed",
         "--config", "configs/temporal.yaml", "--data-root", str(SYN), "--overwrite"])
    run([py, "-m", "deepfake_detection.train_temporal", "train",
         "--config", "configs/temporal.yaml"])

    # generalization eval
    run([py, "-m", "deepfake_detection.analyze_generalization",
         "--config", "configs/spatial.yaml",
         "--ckpt", "checkpoints/spatial_effb0.pt",
         "--data-root", str(SYN), "--max-frames", "100"])

    # artifacts (spectral only; gradcam optional)
    run([py, "-m", "deepfake_detection.analyze_artifacts",
         "--data-root", str(SYN), "--max-images", "30",
         "--ckpt", "checkpoints/spatial_effb0.pt"])

    print("\nSMOKE TEST PASSED ✅")


if __name__ == "__main__":
    main()
