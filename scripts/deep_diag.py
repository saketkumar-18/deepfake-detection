"""Deep diagnostic: score (a) held-out CREMA-D clips the model NEVER trained on,
(b) all unidpro originals, (c) the svara demo wavs, with the deployed ONNX."""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from deepfake_detection.train_audio import load_wav_ffmpeg, segments_from_clip  # noqa: E402

sess = ort.InferenceSession(str(ROOT / "app/assets/model.audio.onnx"), providers=["CPUExecutionProvider"])


def score(p: Path) -> float:
    try:
        x = load_wav_ffmpeg(p)
    except Exception as e:
        return float("nan")
    segs = list(segments_from_clip(x, 32000, False, None))
    if not segs:
        return float("nan")
    X = np.stack(segs)[:, None, :].astype(np.float32)
    logits = sess.run(None, {"waveform": X})[0]
    return float((1 / (1 + np.exp(-logits.squeeze(-1)))).mean())


# (a) held-out CREMA-D: training used wavs[:300] of the SORTED list; 301+ never seen
cremad = sorted((ROOT / "data/audio_cremad").glob("*.wav"))
held = cremad[300:400]
probs = [score(p) for p in held]
probs = [p for p in probs if not np.isnan(p)]
print(f"HELD-OUT CREMA-D (n={len(probs)}, never trained): mean={np.mean(probs):.3f} "
      f"median={np.median(probs):.3f} p90={np.percentile(probs, 90):.3f}")
print("  first 10:", " ".join(f"{p:.2f}" for p in probs[:10]))

# (b) all unidpro originals
orig = sorted((ROOT / "data/audio_unidpro").glob("*original*"))
oprobs = [score(p) for p in orig]
print(f"UNIDPRO ORIGINALS (n={len(oprobs)}): mean={np.mean(oprobs):.3f} min={np.min(oprobs):.3f} max={np.max(oprobs):.3f}")
print("  all:", " ".join(f"{p:.2f}" for p in oprobs))

# (c) svara demo wavs
sv = sorted((Path.home() / "Downloads/svara/app/samples").glob("*.wav"))
print("SVARA SAMPLES:", " ".join(f"{p.name.split('.')[0]}={score(p):.2f}" for p in sv))
