"""Hugging Face Space entrypoint.

Downloads model checkpoints from GitHub releases on cold start, then
launches the Gradio UI. Checkpoints are attached to the repo release
(spatial_effb0.pt, temporal_transformer.pt).
"""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

RELEASE_BASE = (
    "https://github.com/saketkumar-18/deepfake-detection/releases/latest/download"
)
CKPT_DIR = Path(__file__).parent / "checkpoints"


def fetch(name: str) -> Path | None:
    dst = CKPT_DIR / name
    if dst.exists() and dst.stat().st_size > 1_000_000:
        return dst
    CKPT_DIR.mkdir(exist_ok=True)
    url = f"{RELEASE_BASE}/{name}"
    try:
        print(f"[space] downloading {url}")
        urllib.request.urlretrieve(url, dst)
        return dst
    except Exception as e:  # noqa: BLE001
        print(f"[space] could not fetch {name}: {e}")
        return None


# point the package at the downloaded checkpoints before importing the app
os.environ.setdefault("DFD_SPATIAL_CKPT", str(fetch("spatial_effb0.pt") or ""))
os.environ.setdefault("DFD_TEMPORAL_CKPT", str(fetch("temporal_transformer.pt") or ""))

from deepfake_detection.app import build_ui  # noqa: E402

demo = build_ui()
demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
