"""Cross-generator generalization study.

Protocol (the capstone's research question: which artifacts generalize?):
  A. Intra-generator:  train on generator G, test on G.
  B. Cross-generator:  train on generator G, test on every other generator.
  C. Mixed -> held-out: train on a mix of generators, test on the held-out one.
  D. Cross-dataset:    train on FF++ mix, test on CelebDF frames.

Because full retraining per fold is expensive on CPU, this script supports
two modes:
  - 'eval' mode (default): use ONE trained spatial detector and report
    per-generator AUC on every generator folder present in the data root.
    This answers: does a detector trained on the mix generalize to each
    generator, and to CelebDF?
  - 'matrix' mode: run leave-one-generator-out training (needs --train).

Usage:
    python -m deepfake_detection.analyze_generalization \
        --config configs/spatial.yaml --ckpt checkpoints/spatial_effb0.pt \
        --data-root data/processed
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import load_config, resolve_path
from .dataset import FrameDataset, build_eval_transform, scan_frame_tree
from .metrics import binary_metrics
from .models.spatial import load_spatial
from .utils import get_device, set_seed


@torch.no_grad()
def score_generator(model, loader, device):
    scores, labels = [], []
    for imgs, labs, _g, _v in loader:
        logits = model(imgs.to(device)).squeeze(-1)
        scores.extend(torch.sigmoid(logits).cpu().tolist())
        labels.extend(labs.tolist())
    return np.asarray(labels), np.asarray(scores)


def run(cfg: dict, args) -> dict:
    set_seed(42)
    device = get_device()
    model = load_spatial(args.ckpt).to(device).eval()
    img_size = torch.load(args.ckpt, map_location="cpu")["config"].get("img_size", 224)
    transform = build_eval_transform(img_size)

    data_root = Path(args.data_root) if args.data_root else resolve_path(cfg, "data", "processed_root")
    samples = scan_frame_tree(data_root)
    print(f"[generalization] {len(samples)} frames under {data_root}")

    # bucket frames by generator
    by_gen: dict[str, list] = defaultdict(list)
    for s in samples:
        by_gen[s.generator].append(s)

    real_samples = by_gen.get("real", [])
    if not real_samples:
        raise SystemExit("No 'real' frames found; cannot compute AUC.")

    # source-aware real pools (filename prefixes from prepare_data)
    celeb_reals = [s for s in real_samples if s.video_id.startswith("celeb_real")]
    ffpp_reals = [s for s in real_samples if s.video_id.startswith("ffpp_real")]

    max_frames = args.max_frames
    results = {}
    for gen, gen_samples in sorted(by_gen.items()):
        if gen == "real":
            continue
        # match real pool to the generator's source when possible
        if gen == "CelebDF" and celeb_reals:
            reals_pool = celeb_reals
        elif gen in ("FF++", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter") and ffpp_reals:
            reals_pool = ffpp_reals
        else:
            reals_pool = real_samples
        # subsample for speed
        pool = gen_samples[:max_frames]
        reals = reals_pool[:max_frames]
        eval_samples = pool + reals
        ds = FrameDataset(eval_samples, transform)
        loader = DataLoader(ds, batch_size=cfg["eval"]["batch_size"], shuffle=False, num_workers=2)
        labels, scores = score_generator(model, loader, device)
        m = binary_metrics(labels, scores)
        results[gen] = m
        print(f"  {gen:20s} AUC={m['auc']:.4f}  acc={m['acc']:.4f}  n={m['n']}")

    # overall
    all_samples = [s for g, ss in by_gen.items() for s in ss[:max_frames]]
    ds = FrameDataset(all_samples, transform)
    loader = DataLoader(ds, batch_size=cfg["eval"]["batch_size"], shuffle=False, num_workers=2)
    labels, scores = score_generator(model, loader, device)
    results["overall"] = binary_metrics(labels, scores)
    print(f"  {'OVERALL':20s} AUC={results['overall']['auc']:.4f}")

    out = Path(args.out) if args.out else Path("reports/generalization.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"[generalization] saved -> {out}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/spatial.yaml")
    ap.add_argument("--ckpt", default="checkpoints/spatial_effb0.pt")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--max-frames", type=int, default=2000, help="cap frames per generator for eval speed")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run(cfg, args)


if __name__ == "__main__":
    main()
