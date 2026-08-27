"""Hugging Face Space entrypoint.

Downloads model checkpoints from the GitHub repo releases (or bundled
checkpoints/) on cold start, then launches the Gradio UI.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the package is importable when deployed as a Space
sys.path.insert(0, str(Path(__file__).parent))

from deepfake_detection.app import build_ui  # noqa: E402

demo = build_ui()
demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
