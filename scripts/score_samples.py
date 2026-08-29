"""Score the 4 sample videos' audio tracks through the exported ONNX audio model
(the exact runtime the browser demo uses)."""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deepfake_detection.train_audio import load_wav_ffmpeg, segments_from_clip  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sess = ort.InferenceSession(str(ROOT / "app" / "assets" / "model.audio.onnx"),
                            providers=["CPUExecutionProvider"])

for f in sorted((ROOT / "app" / "samples").glob("*.mp4")):
    # extract raw audio via ffmpeg to f32 wav data
    import subprocess
    cmd = ["ffmpeg", "-v", "error", "-i", str(f), "-ac", "1", "-ar", "16000", "-f", "f32le", "-"]
    x = np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, dtype=np.float32)
    segs = list(segments_from_clip(x, 32000, False, None))
    if not segs:
        print(f"{f.name}: NO AUDIO")
        continue
    X = np.stack(segs)[:, None, :].astype(np.float32)
    logits = sess.run(None, {"waveform": X})[0]
    probs = 1 / (1 + np.exp(-logits.squeeze(-1)))
    print(f"{f.name}: audio_fake_prob={probs.mean():.3f}  ({len(segs)} segs)")
