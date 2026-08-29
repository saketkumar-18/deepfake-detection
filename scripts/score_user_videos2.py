"""Score user videos through the SITE-equivalent pipeline: Haar face crops
(160px, margin 0.35) + visual ONNX, plus audio ONNX — and fused verdicts."""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from deepfake_detection.train_audio import load_wav_ffmpeg, segments_from_clip  # noqa: E402

W_V = 0.65
vid_sess = ort.InferenceSession(str(ROOT / "app/assets/model.fp16.onnx"), providers=["CPUExecutionProvider"])
aud_sess = ort.InferenceSession(str(ROOT / "app/assets/model.audio.onnx"), providers=["CPUExecutionProvider"])


def visual_score(path: Path) -> tuple[float, int] | None:
    cap = cv2.VideoCapture(str(path))
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    idxs = [int((i + 0.5) / 16 * max(total - 1, 0)) for i in range(16)]
    probs = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        crop = None
        from deepfake_detection.preprocess import crop_face
        crop = crop_face(frame, img_size=160, margin=0.35, min_face=48)
        if crop is None:
            continue
        x = crop[:, :, ::-1].astype(np.float32) / 255.0
        x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        X = x.transpose(2, 0, 1)[None].astype(np.float32)
        p = vid_sess.run(None, {"image": X})[0].reshape(-1)[0]
        probs.append(float(p))
    cap.release()
    if not probs:
        return None
    return float(np.mean(probs)), len(probs)


def audio_score(path: Path):
    try:
        cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "f32le", "-"]
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        x = np.frombuffer(raw, dtype=np.float32)
        segs = list(segments_from_clip(x, 32000, False, None))
        n = 32000
        rem = len(x) % n
        if not segs and rem > n * 0.25:
            seg = np.zeros(n, dtype=np.float32)
            seg[:rem] = x[-rem:]
            segs = [seg]
        if not segs:
            return None
        X = np.stack(segs)[:, None, :].astype(np.float32)
        probs = aud_sess.run(None, {"waveform": X})[0]  # calibrated probabilities
        return float(probs.mean()), len(segs)
    except Exception:
        return None


def verdict(p):
    if p >= 0.85: return "LIKELY FAKE"
    if p >= 0.60: return "SUSPICIOUS"
    if p >= 0.35: return "UNCERTAIN"
    return "LIKELY REAL"


for f in sorted((ROOT / "sample video").glob("*.mp4")):
    v = visual_score(f)
    a = audio_score(f)
    if v is None:
        print(f"{f.name[:48]:48}  no face crops")
        continue
    pv, nfaces = v
    pa = a[0] if a else None
    if pa is None:
        fused, mode = pv, "visual-only"
    else:
        fused = W_V * pv + (1 - W_V) * pa
        hi = max(pv, pa)
        if hi >= 0.9 and fused < hi:
            fused = hi
        mode = "fused"
    print(f"{f.name[:48]:48}  visual={pv:.3f} ({nfaces} faces)  audio={f'{pa:.3f} ({a[1]}seg)' if a else 'none'}  -> {fused:.3f} {verdict(fused)} [{mode}]")
