"""Extract the actual face crops the browser demo would detect+score from a
user video, and score each with the deployed visual ONNX. Saves crops as images
so we can see WHAT the model is looking at."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

sess = ort.InferenceSession(str(ROOT / "app/assets/model.fp16.onnx"), providers=["CPUExecutionProvider"])

video = sys.argv[1] if len(sys.argv) > 1 else "sample video/WhatsApp Video 2026-08-28 at 22.23.16.mp4"
out_dir = ROOT / "debug_crops"
out_dir.mkdir(exist_ok=True)

cap_frames = subprocess.run(
    ["ffmpeg", "-y", "-v", "error", "-i", str(ROOT / video),
     "-vf", "fps=0.5",
     "-frames:v", "6", str(out_dir / "frame_%02d.png")],
    capture_output=True)
print("extract rc:", cap_frames.returncode, cap_frames.stderr.decode()[:200] if cap_frames.returncode else "ok")

from deepfake_detection.preprocess import crop_face  # reuse face cropper
import cv2

for f in sorted(out_dir.glob("frame_*.png")):
    img = cv2.imread(str(f))
    crop = crop_face(img, img_size=160, margin=0.35, min_face=48)
    if crop is None:
        print(f"{f.name}: NO FACE")
        continue
    crop160 = cv2.resize(crop, (160, 160))[:, :, ::-1].astype(np.float32) / 255.0
    crop160 = (crop160 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    X = crop160.transpose(2, 0, 1)[None].astype(np.float32)
    p = sess.run(None, {"image": X})[0].reshape(-1)[0]
    cv2.imwrite(str(out_dir / f"crop_{f.stem}.png"), crop)
    print(f"{f.name}: face found, visual fake_prob={p:.3f}  (crop saved)")
