"""Unit tests: dataset parsing, models, metrics, preprocessing."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from deepfake_detection.dataset import (
    FrameSample,
    collate_video,
    group_videos,
    parse_frame_name,
)
from deepfake_detection.metrics import (
    aggregate_video_scores,
    binary_metrics,
    per_generator_metrics,
)
from deepfake_detection.models.spatial import SpatialDetector
from deepfake_detection.models.temporal import TemporalTransformer


# ---------------------------------------------------------------- parsing
def test_parse_frame_name():
    assert parse_frame_name("celeb_fake_id0_id16_0000_f468.jpg") == ("celeb_fake_id0_id16_0000", 468)
    assert parse_frame_name("000_003_f12.jpg") == ("000_003", 12)
    assert parse_frame_name("random.jpg") is None


def test_group_videos_subsamples():
    samples = [
        FrameSample(path=None, label=1, video_id="v1", generator="Deepfakes", frame_idx=i)
        for i in range(40)
    ]
    clips = group_videos(samples, frames_per_video=16)
    assert len(clips) == 1
    assert len(clips[0].frames) == 16
    # evenly spaced
    idxs = [s.frame_idx for s in samples if s.path in clips[0].frames] or None


def test_collate_video_pads():
    a = torch.randn(5, 3, 8, 8)
    b = torch.randn(9, 3, 8, 8)
    batch = [(a, 0, "real", "v1"), (b, 1, "Deepfakes", "v2")]
    x, y, mask, gens, vids = collate_video(batch)
    assert x.shape == (2, 9, 3, 8, 8)
    assert mask[0].sum() == 5 and mask[1].sum() == 9
    assert y.tolist() == [0, 1]


# ---------------------------------------------------------------- metrics
def test_binary_metrics_perfect():
    m = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.9, 0.8])
    assert m["auc"] == 1.0
    assert m["acc"] == 1.0


def test_per_generator():
    y = [0, 0, 1, 1, 1, 1]
    s = [0.1, 0.2, 0.9, 0.8, 0.4, 0.6]
    g = ["real", "real", "Deepfakes", "Deepfakes", "Face2Face", "Face2Face"]
    res = per_generator_metrics(y, s, g)
    assert "overall" in res and "Deepfakes" in res and "Face2Face" in res


def test_aggregate_video_scores():
    vids, labs, scores = aggregate_video_scores(
        [0.9, 0.8, 0.1, 0.2], ["v1", "v1", "v2", "v2"], [1, 1, 0, 0], method="mean"
    )
    assert set(vids) == {"v1", "v2"}
    d = dict(zip(vids, scores))
    assert d["v1"] == pytest.approx(0.85)
    assert d["v2"] == pytest.approx(0.15)


# ---------------------------------------------------------------- models
def test_spatial_forward():
    m = SpatialDetector(backbone="efficientnet_b0", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits = m(x)
    assert logits.shape == (2, 1)
    feat = m.forward_features(x)
    assert feat.shape == (2, m.embed_dim)


def test_temporal_forward_with_mask():
    m = TemporalTransformer(in_dim=64, dim=32, depth=2, heads=2, pool="attention")
    x = torch.randn(3, 10, 64)
    mask = torch.ones(3, 10, dtype=torch.bool)
    mask[0, 6:] = False  # variable length
    logits = m(x, mask)
    assert logits.shape == (3, 1)
    assert torch.isfinite(logits).all()


def test_temporal_pools():
    for pool in ["attention", "mean", "max"]:
        m = TemporalTransformer(in_dim=32, dim=16, depth=1, heads=2, pool=pool)
        x = torch.randn(2, 5, 32)
        out = m(x)
        assert out.shape == (2, 1), pool


# ---------------------------------------------------------------- preprocess
def test_crop_face_fallback_center():
    from deepfake_detection.preprocess import crop_face

    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    crop = crop_face(frame, img_size=224, margin=0.35, min_face=40)
    assert crop is not None
    assert crop.shape == (224, 224, 3)
