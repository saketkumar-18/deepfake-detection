"""Examine the one flagged clip's audio: spectrogram stats + segment scores.
Compare with a known-real WhatsApp clip."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from deepfake_detection.train_audio import load_wav_ffmpeg, segments_from_clip  # noqa: E402

sess = ort.InferenceSession(str(ROOT / "app/assets/model.audio.onnx"), providers=["CPUExecutionProvider"])


def report(name, path):
    x = load_wav_ffmpeg(ROOT / path)
    # energy, spectral flatness proxy (std of spectrum), zero-crossing rate
    zcr = float(np.mean(np.abs(np.diff(np.sign(x))) > 0))
    rms = float(np.sqrt(np.mean(x ** 2)))
    spec = np.abs(np.fft.rfft(x[: 2 ** 15]))
    flat = float(np.std(np.log(spec[1:] + 1e-9)))
    segs = list(segments_from_clip(x, 32000, False, None))
    X = np.stack(segs)[:, None, :].astype(np.float32)
    probs = np.asarray(sess.run(None, {"waveform": X})[0]).reshape(-1)
    print(f"{name}:")
    print(f"  dur={len(x)/16000:.1f}s rms={rms:.3f} zcr={zcr:.3f} log-spec-std={flat:.3f}")
    print(f"  per-seg probs: {' '.join(f'{p:.2f}' for p in probs)}")


report("REAL?  22.23.17 (flagged 1.000)", "sample video/WhatsApp Video 2026-08-28 at 22.23.17.mp4")
report("REAL   22.23.16 (scores 0.000)", "sample video/WhatsApp Video 2026-08-28 at 22.23.16.mp4")
report("SYNTH  unidpro synthetic_2 (should be 1.0)", "data/audio_unidpro/UK_female_1_synthetic_2.mp3")
