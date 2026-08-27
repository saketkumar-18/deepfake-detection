"""Evaluation metrics: frame-level and video-level AUC, per-generator breakdown."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


def binary_metrics(y_true, y_score):
    """Return AUC, AP, accuracy at 0.5 threshold for binary scores."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= 0.5).astype(int)
    out = {
        "auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        "ap": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        "acc": float(accuracy_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }
    return out


def per_generator_metrics(y_true, y_score, generators):
    """Compute AUC overall and per generator (fake generators vs. pooled real)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    gens = np.asarray(generators)

    results = {"overall": binary_metrics(y_true, y_score)}
    real_mask = gens == "real"
    for g in sorted(set(gens) - {"real"}):
        mask = real_mask | (gens == g)
        if mask.sum() < 2:
            continue
        results[g] = binary_metrics(y_true[mask], y_score[mask])
    return results


def aggregate_video_scores(frame_scores, video_ids, labels, method="mean"):
    """Aggregate frame-level fake-probabilities to video-level.

    Returns (video_ids, video_labels, video_scores).
    """
    by_vid: dict[str, list[float]] = defaultdict(list)
    vid_label: dict[str, int] = {}
    for vid, lab, sc in zip(video_ids, labels, frame_scores):
        by_vid[vid].append(float(sc))
        vid_label[vid] = int(lab)
    vids, labs, scores = [], [], []
    for vid, scs in by_vid.items():
        arr = np.asarray(scs)
        if method == "mean":
            agg = float(arr.mean())
        elif method == "max":
            agg = float(arr.max())
        elif method == "topk":
            k = max(1, len(arr) // 4)
            agg = float(np.sort(arr)[-k:].mean())
        else:
            agg = float(arr.mean())
        vids.append(vid)
        labs.append(vid_label[vid])
        scores.append(agg)
    return vids, labs, scores
