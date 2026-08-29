"""Sanity check: audio model on out-of-domain natural speech (should be LOW fake-prob)
and on unidpro synthetic clips (should be HIGH). Prints clip-level probabilities."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deepfake_detection.train_audio import AudioCNN, load_wav_ffmpeg, segments_from_clip  # noqa: E402

CKPT = Path(__file__).resolve().parents[1] / "checkpoints" / "audio_cnn.pt"
model = AudioCNN()
sd = torch.load(CKPT, map_location="cpu")
model.load_state_dict(sd["model"])
model.eval()


def score(path: Path) -> float:
    x = load_wav_ffmpeg(path)
    segs = list(segments_from_clip(x, 32000, False, None))
    X = torch.from_numpy(np.stack(segs)).unsqueeze(1)
    with torch.no_grad():
        p = torch.sigmoid(model(X)).squeeze(-1).numpy()
    return float(p.mean())


print("=== out-of-domain NATURAL speech (svara/CREMA-D) — expect LOW ===")
sv = sorted((Path.home() / "Downloads" / "svara" / "app" / "samples").glob("*.wav"))
for f in sv[:4]:
    print(f"  {f.name}: {score(f):.3f}")

print("=== unidpro SYNTHETIC — expect HIGH ===")
un = sorted((Path(__file__).resolve().parents[1] / "data" / "audio_unidpro").glob("*synthetic*"))
for f in un[:4]:
    print(f"  {f.name}: {score(f):.3f}")

print("=== unidpro ORIGINAL — expect LOW ===")
uo = sorted((Path(__file__).resolve().parents[1] / "data" / "audio_unidpro").glob("*original*"))
for f in uo[:4]:
    print(f"  {f.name}: {score(f):.3f}")
