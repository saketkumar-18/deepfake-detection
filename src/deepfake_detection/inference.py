"""End-to-end inference: video -> fake probability + explanation.

Combines the spatial detector (frame scores) and the temporal transformer
(video-level aggregation). Falls back to spatial-only aggregation if no
temporal checkpoint is present.

Usage:
    python -m deepfake_detection.inference path/to/video.mp4 \
        [--spatial checkpoints/spatial_effb0.pt] \
        [--temporal checkpoints/temporal_transformer.pt]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .config import PROJECT_ROOT
from .dataset import build_eval_transform
from .models.spatial import load_spatial
from .models.temporal import TemporalTransformer
from .preprocess import crop_face, sample_frames


class DeepfakeVideoDetector:
    """Facade for video-level deepfake scoring."""

    def __init__(
        self,
        spatial_ckpt: str | Path,
        temporal_ckpt: str | Path | None = None,
        frames_per_video: int = 16,
        device: str | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.frames_per_video = frames_per_video
        self.spatial = load_spatial(spatial_ckpt).to(self.device).eval()
        self.img_size = torch.load(spatial_ckpt, map_location="cpu")["config"].get("img_size", 224)
        self.transform = build_eval_transform(self.img_size)

        self.temporal = None
        if temporal_ckpt is not None and Path(temporal_ckpt).exists():
            ck = torch.load(temporal_ckpt, map_location="cpu")
            mcfg = ck["config"]
            self.temporal = TemporalTransformer(
                in_dim=mcfg["in_dim"],
                dim=mcfg["dim"],
                depth=mcfg["depth"],
                heads=mcfg["heads"],
                mlp_ratio=mcfg.get("mlp_ratio", 2.0),
                dropout=mcfg.get("dropout", 0.1),
                pool=mcfg.get("pool", "attention"),
            ).to(self.device)
            self.temporal.load_state_dict(ck["model"])
            self.temporal.eval()

    @torch.no_grad()
    def predict_video(self, video_path: str | Path) -> dict:
        """Return dict with video-level fake probability and frame scores."""
        video_path = Path(video_path)
        frames = sample_frames(video_path, self.frames_per_video)
        if not frames:
            return {"error": f"could not read frames from {video_path}"}

        crops = []
        for f in frames:
            c = crop_face(f, self.img_size, margin=0.35, min_face=40)
            crops.append(c)

        tensors = torch.stack([self.transform(_to_pil(c)) for c in crops]).to(self.device)

        # spatial frame scores + embeddings
        frame_scores, embs = [], []
        for i in range(0, tensors.shape[0], 16):
            x = tensors[i : i + 16]
            feat = self.spatial.forward_features(x)
            logits = self.spatial.head(feat).squeeze(-1)
            frame_scores.extend(torch.sigmoid(logits).cpu().tolist())
            embs.append(feat.cpu())
        emb = torch.cat(embs, 0)

        spatial_score = float(np.mean(frame_scores))
        result = {
            "video": str(video_path),
            "frames_analyzed": len(frame_scores),
            "frame_scores": [round(s, 4) for s in frame_scores],
            "spatial_score": round(spatial_score, 4),
        }

        if self.temporal is not None:
            x = emb.unsqueeze(0).to(self.device)
            mask = torch.ones(1, x.shape[1], dtype=torch.bool, device=self.device)
            logit = self.temporal(x, mask).squeeze()
            result["temporal_score"] = round(float(torch.sigmoid(logit)), 4)
            result["score"] = result["temporal_score"]
            result["method"] = "spatial+temporal"
        else:
            result["score"] = spatial_score
            result["method"] = "spatial-only (mean aggregation)"

        result["verdict"] = verdict(result["score"])
        return result


def _to_pil(arr):
    from PIL import Image

    return Image.fromarray(arr)


def verdict(score: float) -> str:
    if score >= 0.75:
        return "LIKELY FAKE"
    if score >= 0.5:
        return "SUSPICIOUS"
    if score >= 0.25:
        return "UNCERTAIN"
    return "LIKELY REAL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--spatial", default="checkpoints/spatial_effb0.pt")
    ap.add_argument("--temporal", default="checkpoints/temporal_transformer.pt")
    args = ap.parse_args()

    spatial = args.spatial if Path(args.spatial).is_absolute() else PROJECT_ROOT / args.spatial
    temporal = args.temporal if Path(args.temporal).is_absolute() else PROJECT_ROOT / args.temporal
    det = DeepfakeVideoDetector(spatial, temporal)
    res = det.predict_video(args.video)
    import json

    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
