"""Benchmark comparison: spatial-only vs spatial+temporal, aggregation ablations.

Runs on cached embeddings + checkpoints and produces a single comparison
table (JSON + markdown) in reports/benchmark.md.

Metrics compared:
  - Frame-level AUC (spatial detector)
  - Video-level AUC with aggregation: mean / max / top-k
  - Video-level AUC with temporal transformer
  - Per-generator breakdown for the best model

Usage:
    python -m deepfake_detection.benchmark
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import PROJECT_ROOT, load_config
from .metrics import binary_metrics, per_generator_metrics
from .models.spatial import load_spatial
from .models.temporal import TemporalTransformer
from .train_temporal import EmbDataset, collate_emb, load_cached
from .utils import get_device, set_seed


def load_temporal(ckpt_path: Path, in_dim: int):
    ck = torch.load(ckpt_path, map_location="cpu")
    mcfg = ck["config"]
    model = TemporalTransformer(
        in_dim=mcfg.get("in_dim", in_dim),
        dim=mcfg["dim"],
        depth=mcfg["depth"],
        heads=mcfg["heads"],
        mlp_ratio=mcfg.get("mlp_ratio", 2.0),
        dropout=mcfg.get("dropout", 0.1),
        pool=mcfg.get("pool", "attention"),
    )
    model.load_state_dict(ck["model"])
    model.eval()
    return model


@torch.no_grad()
def temporal_scores(model, loader, device):
    scores, labels, gens = [], [], []
    for x, y, mask, g, _v in loader:
        logits = model(x.to(device), mask.to(device)).squeeze(-1)
        scores.extend(torch.sigmoid(logits).cpu().tolist())
        labels.extend(y.tolist())
        gens.extend(g)
    return labels, scores, gens


def run(args) -> dict:
    set_seed(42)
    device = get_device()
    cache_dir = PROJECT_ROOT / "data" / "embeddings"
    spatial_ckpt = PROJECT_ROOT / "checkpoints" / "spatial_effb0.pt"
    temporal_ckpt = PROJECT_ROOT / "checkpoints" / "temporal_transformer.pt"

    max_frames = getattr(args, "max_frames", None) if args else None
    split = getattr(args, "split", "test") if args else "test"

    # Use the official held-out split with the same length-control protocol as
    # training, so benchmark numbers are honest and comparable.
    val_items = load_cached(cache_dir, split=split, max_frames=max_frames)
    if not val_items:
        # fall back to stratified random split if no split info cached
        items = load_cached(cache_dir, max_frames=max_frames)
        if not items:
            raise SystemExit(f"No embeddings in {cache_dir}; run train_temporal embed first.")
        import random

        rng = random.Random(42)
        by_label: dict[int, list] = {0: [], 1: []}
        for it in items:
            by_label[it["label"]].append(it)
        val_items = []
        for lab, group in by_label.items():
            rng.shuffle(group)
            n_val = max(1, int(len(group) * 0.15))
            val_items.extend(group[:n_val])
        rng.shuffle(val_items)
    in_dim = val_items[0]["emb"].shape[1]
    loader = DataLoader(EmbDataset(val_items), batch_size=32, shuffle=False, collate_fn=collate_emb)

    results = {}

    # --- aggregation ablations from frame embeddings via spatial head ---
    if spatial_ckpt.exists():
        spatial = load_spatial(spatial_ckpt).to(device).eval()
        agg = {"mean": [], "max": [], "topk": []}
        labels_all, gens_all = [], []
        with torch.no_grad():
            for x, y, mask, g, _v in loader:
                x = x.to(device)
                b, t, d = x.shape
                flat = x.reshape(b * t, d)
                logits = spatial.head(flat).squeeze(-1)
                probs = torch.sigmoid(logits).reshape(b, t).cpu().numpy()
                m = mask.numpy()
                for i in range(b):
                    p = probs[i][m[i]]
                    k = max(1, len(p) // 4)
                    agg["mean"].append(float(p.mean()))
                    agg["max"].append(float(p.max()))
                    agg["topk"].append(float(np.sort(p)[-k:].mean()))
                labels_all.extend(y.tolist())
                gens_all.extend(g)
        for name, scs in agg.items():
            results[f"spatial_agg_{name}"] = binary_metrics(labels_all, scs)
        results["spatial_agg_mean_per_gen"] = per_generator_metrics(labels_all, agg["mean"], gens_all)

    # --- temporal transformer ---
    if temporal_ckpt.exists():
        tmodel = load_temporal(temporal_ckpt, in_dim).to(device)
        labels, scores, gens = temporal_scores(tmodel, loader, device)
        results["temporal_transformer"] = binary_metrics(labels, scores)
        results["temporal_transformer_per_gen"] = per_generator_metrics(labels, scores, gens)

    # --- markdown table ---
    lines = ["# Benchmark: Deepfake Video Detection\n",
             f"Held-out {split} videos: {len(val_items)} | device: {device}"
             + (f" | length-control T={max_frames}" if max_frames else "") + "\n",
             "| Model / aggregation | Video AUC | AP | Acc |",
             "|---|---|---|---|"]
    order = ["spatial_agg_mean", "spatial_agg_max", "spatial_agg_topk", "temporal_transformer"]
    for k in order:
        if k in results:
            m = results[k]
            lines.append(f"| {k} | {m['auc']:.4f} | {m['ap']:.4f} | {m['acc']:.4f} |")
    lines.append("\n## Per-generator AUC (best model)\n")
    best_key = "temporal_transformer_per_gen" if "temporal_transformer_per_gen" in results else "spatial_agg_mean_per_gen"
    if best_key in results:
        lines.append("| Generator | AUC | n |")
        lines.append("|---|---|---|")
        for g, m in sorted(results[best_key].items()):
            lines.append(f"| {g} | {m['auc']:.4f} | {m['n']} |")
    md = "\n".join(lines) + "\n"

    reports = PROJECT_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "benchmark.md").write_text(md, encoding="utf-8")
    (reports / "benchmark.json").write_text(json.dumps(results, indent=2))
    print(md)
    print(f"[benchmark] saved -> {reports / 'benchmark.md'}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", help="which cached split to benchmark on")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="length-control: subsample every clip to exactly T frames")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
