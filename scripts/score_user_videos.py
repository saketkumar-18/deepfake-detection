"""Score the user's own test videos through BOTH branches (Python ONNX runtime,
the same models deployed to the browser demo) and show the fused verdict."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from deepfake_detection.train_audio import load_wav_ffmpeg, segments_from_clip  # noqa: E402

W_V = 0.65
vid_sess = ort.InferenceSession(str(ROOT / "app/assets/model.fp16.onnx"), providers=["CPUExecutionProvider"])
aud_sess = ort.InferenceSession(str(ROOT / "app/assets/model.audio.onnx"), providers=["CPUExecutionProvider"])


def visual_score(path: Path) -> float | None:
    """Frame-level visual branch via ffmpeg frame extraction + ONNX (mirrors browser)."""
    import cv2
    from deepfake_detection.models.spatial import load_spatial  # not used; ONNX only
    cap = cv2.VideoCapture(str(path))
    frames, n = [], 0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    idxs = [int((i + 0.5) / 16 * total) for i in range(16)] if total > 16 else list(range(int(total)))
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.resize(frame, (160, 160))[:, :, ::-1].astype(np.float32) / 255.0
        frame = (frame - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        frames.append(frame.transpose(2, 0, 1))
    cap.release()
    if not frames:
        return None
    X = np.stack(frames).astype(np.float32)
    probs = []
    for i in range(0, len(X), 8):
        out = vid_sess.run(None, {"image": X[i:i+8]})[0]  # already a probability
        probs.extend(out.reshape(-1).tolist())
    return float(np.mean(probs))


def audio_score(path: Path):
    try:
        cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "f32le", "-"]
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        x = np.frombuffer(raw, dtype=np.float32)
        segs = list(segments_from_clip(x, 32000, False, None))
        # mirror browser padding rule: keep partial final segment if >25%
        n = 32000
        rem = len(x) % n
        if not segs and rem > n * 0.25:
            seg = np.zeros(n, dtype=np.float32)
            seg[:rem] = x[-rem:]
            segs = [seg]
        if not segs:
            return None
        X = np.stack(segs)[:, None, :].astype(np.float32)
        logits = aud_sess.run(None, {"waveform": X})[0]
        probs = 1 / (1 + np.exp(-logits.squeeze(-1)))
        return float(probs.mean()), len(segs)
    except Exception:
        return None


def fuse(pv, pa):
    if pa is None:
        return pv, "visual-only (no audio track)"
    f = W_V * pv + (1 - W_V) * pa
    hi = max(pv, pa)
    if hi >= 0.9 and f < hi:
        f = hi
    return f, "visual+audio fused"


for f in sorted((ROOT / "sample video").glob("*.mp4")):
    try:
        pv = visual_score(f)
    except Exception as e:
        pv = None
        print(f"  (visual error {e})")
    pa = audio_score(f)
    pa_s = f"{pa[0]:.3f} ({pa[1]} segs)" if pa else "NO AUDIO TRACK"
    pv_s = f"{pv:.3f}" if pv is not None else "n/a"
    if pv is not None:
        fused, mode = fuse(pv, pa[0] if pa else None)
        print(f"{f.name[:52]:52}  visual={pv_s}  audio={pa_s}  -> fused={fused:.3f}  [{mode}]")
    else:
        print(f"{f.name[:52]:52}  visual=n/a  audio={pa_s}")
