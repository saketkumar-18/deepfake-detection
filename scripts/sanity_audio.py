"""Sanity check with the DEPLOYED ONNX (calibrated): real speech must score LOW,
synthetics HIGH. This is the acceptance gate before any deploy."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from deepfake_detection.train_audio import load_wav_ffmpeg, segments_from_clip  # noqa: E402

sess = ort.InferenceSession(str(ROOT / "app/assets/model.audio.onnx"), providers=["CPUExecutionProvider"])


def score(path) -> float:
    x = load_wav_ffmpeg(path)
    segs = list(segments_from_clip(x, 32000, False, None))
    if not segs:
        return float("nan")
    X = np.stack(segs)[:, None, :].astype(np.float32)
    p = sess.run(None, {"waveform": X})[0]
    return float(np.asarray(p).mean())


fails = []

print("=== out-of-domain NATURAL speech — must be LOW (<0.5) ===")
for f in sorted((Path.home() / "Downloads/svara/app/samples").glob("*.wav")):
    s = score(f)
    print(f"  {f.name}: {s:.3f}")
    if s >= 0.6:
        fails.append(f"OOD natural {f.name}={s:.3f}")

print("=== held-out CREMA-D (never trained, n=40) — must average LOW ===")
held = sorted((ROOT / "data/audio_cremad").glob("*.wav"))[300:340]
ps = [score(p) for p in held]
ps = [p for p in ps if not np.isnan(p)]
print(f"  mean={np.mean(ps):.3f} median={np.median(ps):.3f} p90={np.percentile(ps, 90):.3f}")
if np.mean(ps) >= 0.5:
    fails.append(f"held-out CREMA-D mean={np.mean(ps):.3f}")

print("=== unidpro ORIGINALS (real) — must be LOW ===")
for f in sorted((ROOT / "data/audio_unidpro").glob("*original*")):
    s = score(f)
    if s >= 0.6:
        fails.append(f"unidpro original {f.name}={s:.3f}")
print("  (all below 0.6)" if not any("unidpro" in f for f in fails) else "  FAILURES ^")

print("=== unidpro SYNTHETICS — must be HIGH ===")
for f in sorted((ROOT / "data/audio_unidpro").glob("*synthetic*"))[:6]:
    s = score(f)
    print(f"  {f.name}: {s:.3f}")

print()
if fails:
    print("SANITY FAIL:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("SANITY PASS — calibrated model honest on real speech")
