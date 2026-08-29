"""Codec-shortcut test: if the audio model detects file format rather than
voice synthesity, transcoding REAL speech to mp3 should flip its score high."""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from deepfake_detection.train_audio import load_wav_ffmpeg, segments_from_clip  # noqa: E402

sess = ort.InferenceSession(str(ROOT / "app/assets/model.audio.onnx"), providers=["CPUExecutionProvider"])


def score_file(p: Path) -> float:
    x = load_wav_ffmpeg(p)
    segs = list(segments_from_clip(x, 32000, False, None))
    if not segs:
        return float("nan")
    X = np.stack(segs)[:, None, :].astype(np.float32)
    logits = sess.run(None, {"waveform": X})[0]
    return float((1 / (1 + np.exp(-logits.squeeze(-1)))).mean())


def transcode(p: Path, codec: str, bitrate: str) -> Path:
    ext = {"libmp3lame": "mp3", "libopus": "ogg", "aac": "m4a"}[codec]
    out = Path(tempfile.mkdtemp()) / f"t.{ext}"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(p), "-c:a", codec,
                    "-b:a", bitrate, str(out)], check=True)
    return out


tests = [
    ("CREMA-D real (native wav)", ROOT / "data/audio_cremad/1001_DFA_ANG_XX.wav"),
    ("CREMA-D real -> mp3 64k", ROOT / "data/audio_cremad/1001_DFA_ANG_XX.wav"),
    ("CREMA-D real -> mp3 128k", ROOT / "data/audio_cremad/1001_DFA_ANG_XX.wav"),
    ("CREMA-D real -> opus 32k", ROOT / "data/audio_cremad/1001_DFA_ANG_XX.wav"),
    ("unidpro original (native m4a)", ROOT / "data/audio_unidpro/UK_female_1_original.m4a"),
    ("unidpro original -> mp3 64k", ROOT / "data/audio_unidpro/UK_female_1_original.m4a"),
    ("unidpro synth (native mp3)", ROOT / "data/audio_unidpro/UK_female_1_synthetic_1.mp3"),
    ("unidpro synth -> m4a aac", ROOT / "data/audio_unidpro/UK_female_1_synthetic_1.mp3"),
    ("WhatsApp real speech", ROOT / "sample video/WhatsApp Video 2026-08-28 at 22.23.16.mp4"),
    ("WhatsApp real -> mp3 64k", ROOT / "sample video/WhatsApp Video 2026-08-28 at 22.23.16.mp4"),
]

for name, src in tests:
    p = src
    if "->" in name:
        spec = name.split("->")[1]  # e.g. " mp3 64k", " opus 32k"
        codec = "libmp3lame" if "mp3" in spec else ("libopus" if "opus" in spec else "aac")
        br = "64k" if "64k" in spec else ("128k" if "128k" in spec else "32k")
        p = transcode(src, codec, br)
    s = score_file(p)
    print(f"{name:36} fake_prob={s:.3f}")
