"""Preprocess raw deepfake videos -> aligned face crops.

Pipeline per video:
  1. sample N frames evenly (ffmpeg via OpenCV),
  2. detect faces (OpenCV Haar cascade; MTCNN-free so no extra weights),
  3. expand + square-crop the face, resize to img_size,
  4. save as frame_%04d.jpg under <out>/<class>/<video_id>/.

Works on Celeb-DF v2 layout (Celeb-real/, Celeb-synthesis/) and FF++ layout
(Real/, Deepfakes/, Face2Face/, ...).

Usage:
    python -m deepfake_detection.preprocess --config configs/preprocess.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .config import load_config, resolve_path

_CASCADE = None


def get_cascade():
    global _CASCADE
    if _CASCADE is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _CASCADE = cv2.CascadeClassifier(path)
    return _CASCADE


def sample_frames(video_path: Path, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    idxs = np.linspace(0, total - 1, n).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def crop_face(frame: np.ndarray, img_size: int, margin: float, min_face: int) -> np.ndarray | None:
    cascade = get_cascade()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_face, min_face))
    h, w = frame.shape[:2]
    if len(faces) == 0:
        # fallback: center crop (faces are usually centered in these datasets)
        s = int(min(h, w) * 0.8)
        y0, x0 = (h - s) // 2, (w - s) // 2
        face = frame[y0 : y0 + s, x0 : x0 + s]
    else:
        # take largest face
        x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
        mx, my = int(fw * margin), int(fh * margin)
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(w, x + fw + mx), min(h, y + fh + my)
        face = frame[y0:y1, x0:x1]
        # square it
        s = max(face.shape[0], face.shape[1])
        pad = np.zeros((s, s, 3), dtype=face.dtype)
        py, px = (s - face.shape[0]) // 2, (s - face.shape[1]) // 2
        pad[py : py + face.shape[0], px : px + face.shape[1]] = face
        face = pad
    return cv2.resize(face, (img_size, img_size), interpolation=cv2.INTER_AREA)


def process_video(video_path: Path, out_dir: Path, cfg: dict) -> int:
    frames = sample_frames(video_path, cfg["frames_per_video"])
    if not frames:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, f in enumerate(frames):
        crop = crop_face(f, cfg["img_size"], cfg["face_margin"], cfg["min_face"])
        if crop is None:
            continue
        out = out_dir / f"frame_{i:04d}.jpg"
        if out.exists() and not cfg.get("overwrite", False):
            saved += 1
            continue
        cv2.imwrite(str(out), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, cfg["jpeg_quality"]])
        saved += 1
    return saved


def run(cfg: dict, args) -> None:
    raw_root = Path(args.raw_root) if args.raw_root else resolve_path(cfg, "raw_root")
    out_root = resolve_path(cfg, "out_root")
    print(f"[preprocess] raw={raw_root} out={out_root}")

    videos = sorted(raw_root.rglob("*.mp4"))
    if args.limit:
        videos = videos[: args.limit]
    print(f"[preprocess] {len(videos)} videos")

    total_saved = 0
    for v in tqdm(videos, desc="videos"):
        rel = v.relative_to(raw_root)
        # keep class dir (first component) + video stem
        parts = rel.parts
        class_dir = parts[0] if len(parts) > 1 else "unknown"
        out_dir = out_root / class_dir / v.stem
        total_saved += process_video(v, out_dir, cfg)
    print(f"[preprocess] saved {total_saved} face crops -> {out_root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/preprocess.yaml")
    ap.add_argument("--raw-root", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run(cfg, args)


if __name__ == "__main__":
    main()
