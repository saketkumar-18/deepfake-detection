"""Is the audio model's real-speech failure arousal-driven?
Score held-out CREMA-D by emotion label (ANG/HAP = high arousal; NEU/SAD = low)."""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from deepfake_detection.train_audio import load_wav_ffmpeg, segments_from_clip  # noqa: E402

sess = ort.InferenceSession(str(ROOT / "app/assets/model.audio.onnx"), providers=["CPUExecutionProvider"])


def score(p):
    x = load_wav_ffmpeg(p)
    segs = list(segments_from_clip(x, 32000, False, None))
    if not segs:
        return float("nan")
    X = np.stack(segs)[:, None, :].astype(np.float32)
    return float(np.asarray(sess.run(None, {"waveform": X})[0]).mean())


held = sorted((ROOT / "data/audio_cremad").glob("*.wav"))[300:420]
by_emo = defaultdict(list)
for p in held:
    emo = p.stem.split("_")[2]  # 1001_DFA_ANG_XX -> ANG
    by_emo[emo].append(score(p))

for emo, ps in sorted(by_emo.items()):
    ps = [p for p in ps if not np.isnan(p)]
    hi = sum(1 for p in ps if p >= 0.6)
    print(f"{emo}: n={len(ps)} mean={np.mean(ps):.3f} high(>=0.6)={hi}/{len(ps)}")
