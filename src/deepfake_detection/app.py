"""Gradio demo app for deepfake video detection.

Run:
    python -m deepfake_detection.app
Then open http://127.0.0.1:7860
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr

from .config import PROJECT_ROOT
from .inference import DeepfakeVideoDetector

SPATIAL = Path(os.environ.get("DFD_SPATIAL_CKPT") or "") or PROJECT_ROOT / "checkpoints" / "spatial_effb0.pt"
TEMPORAL = Path(os.environ.get("DFD_TEMPORAL_CKPT") or "") or PROJECT_ROOT / "checkpoints" / "temporal_transformer.pt"

_detector: DeepfakeVideoDetector | None = None


def get_detector() -> DeepfakeVideoDetector:
    global _detector
    if _detector is None:
        if not SPATIAL.exists():
            raise RuntimeError(f"Spatial checkpoint missing: {SPATIAL}. Train it first.")
        _detector = DeepfakeVideoDetector(
            SPATIAL, TEMPORAL if TEMPORAL.exists() else None
        )
    return _detector


def predict(video_path: str | None):
    if not video_path:
        return "Upload a video first.", "{}"
    det = get_detector()
    res = det.predict_video(video_path)
    if "error" in res:
        return res["error"], "{}"

    score = res["score"]
    verdict = res["verdict"]
    emoji = {"LIKELY FAKE": "🚨", "SUSPICIOUS": "⚠️", "UNCERTAIN": "❓", "LIKELY REAL": "✅"}[verdict]
    bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
    summary = (
        f"{emoji} **{verdict}**\n\n"
        f"Fake probability: **{score:.1%}**\n`{bar}` {score:.3f}\n\n"
        f"Method: {res['method']} · frames analyzed: {res['frames_analyzed']}\n"
        f"Spatial (frame-mean) score: {res['spatial_score']:.3f}"
    )
    if "temporal_score" in res:
        summary += f"\nTemporal (transformer) score: {res['temporal_score']:.3f}"
    return summary, json.dumps(res, indent=2)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Deepfake Video Detector") as demo:
        gr.Markdown(
            "# 🕵️ Deepfake Video Detector\n"
            "Spatial-temporal CNN/Transformer — EfficientNet-B0 frame detector + "
            "temporal transformer aggregation. Trained on FF++ (5 generators) + Celeb-DF v2."
        )
        with gr.Row():
            with gr.Column():
                video = gr.Video(label="Upload a video")
                btn = gr.Button("Analyze", variant="primary")
            with gr.Column():
                out_md = gr.Markdown()
                out_json = gr.JSON(label="Raw output")
        gr.Markdown(
            "*Research prototype — scores are forensic evidence cues, not legal proof. "
            "Short clips (<30s) with a visible face work best.*"
        )
        btn.click(predict, inputs=video, outputs=[out_md, out_json])
    return demo


def main():
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
