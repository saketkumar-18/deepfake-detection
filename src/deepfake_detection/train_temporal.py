"""Train the temporal transformer over frozen spatial embeddings.

Two stages:
  1. embed: run the frozen spatial detector over every frame, cache (T, D)
     embeddings per video to .npy files.
  2. train: train the TemporalTransformer on cached embeddings.

Usage:
    python -m deepfake_detection.train_temporal --config configs/temporal.yaml embed
    python -m deepfake_detection.train_temporal --config configs/temporal.yaml train
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import load_config, resolve_path
from .dataset import (
    VideoDataset,
    build_eval_transform,
    collate_video,
    group_videos,
    scan_frame_tree,
)
from .metrics import binary_metrics, per_generator_metrics
from .models.spatial import load_spatial
from .models.temporal import TemporalTransformer
from .utils import get_device, set_seed


@torch.no_grad()
def embed_all(cfg: dict, args) -> None:
    """Cache per-video frame embeddings from the frozen spatial detector."""
    device = get_device()
    spatial_ckpt = resolve_path(cfg, "embed", "spatial_ckpt")
    data_root = Path(args.data_root) if args.data_root else resolve_path(cfg, "data", "processed_root")
    cache_dir = resolve_path(cfg, "embed", "cache_dir")
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[embed] loading spatial model from {spatial_ckpt}")
    spatial = load_spatial(spatial_ckpt).to(device).eval()

    print(f"[embed] scanning {data_root}")
    samples = scan_frame_tree(data_root)
    frames_per_video = cfg["embed"]["frames_per_video"]
    clips = group_videos(samples, frames_per_video)
    print(f"[embed] {len(clips)} videos")

    img_size = spatial_ckpt and torch.load(spatial_ckpt, map_location="cpu")["config"].get("img_size", 224)
    transform = build_eval_transform(img_size)

    done = 0
    t0 = time.time()
    for clip in clips:
        out = cache_dir / f"{clip.video_id}.npz"
        if out.exists() and not args.overwrite:
            done += 1
            continue
        tensors = []
        ok = True
        for p in clip.frames:
            try:
                from PIL import Image

                img = Image.open(p).convert("RGB")
                tensors.append(transform(img))
            except Exception as e:  # noqa: BLE001
                ok = False
                break
        if not ok or not tensors:
            continue
        x = torch.stack(tensors).to(device)
        feats = []
        for i in range(0, x.shape[0], 32):
            feats.append(spatial.forward_features(x[i : i + 32]).cpu())
        emb = torch.cat(feats, 0).numpy().astype(np.float16)
        np.savez_compressed(out, emb=emb, label=clip.label, generator=clip.generator)
        done += 1
        if done % 50 == 0:
            print(f"  embedded {done}/{len(clips)} ({time.time() - t0:.0f}s)")
    print(f"[embed] done: {done} videos cached -> {cache_dir}")


def load_cached(cache_dir: Path):
    """Load all cached embeddings. Returns list of dicts."""
    items = []
    for f in sorted(cache_dir.glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        items.append(
            {
                "video_id": f.stem,
                "emb": d["emb"].astype(np.float32),
                "label": int(d["label"]),
                "generator": str(d["generator"]),
            }
        )
    return items


class EmbDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        return torch.from_numpy(it["emb"]), it["label"], it["generator"], it["video_id"]


def collate_emb(batch):
    xs, labels, gens, vids = zip(*batch)
    max_t = max(x.shape[0] for x in xs)
    dim = xs[0].shape[1]
    padded = torch.zeros(len(xs), max_t, dim)
    mask = torch.zeros(len(xs), max_t, dtype=torch.bool)
    for i, x in enumerate(xs):
        padded[i, : x.shape[0]] = x
        mask[i, : x.shape[0]] = True
    return padded, torch.tensor(labels, dtype=torch.long), mask, list(gens), list(vids)


@torch.no_grad()
def evaluate_temporal(model, loader, device):
    model.eval()
    scores, labels, gens = [], [], []
    for x, y, mask, g, _v in loader:
        x, mask = x.to(device), mask.to(device)
        logits = model(x, mask).squeeze(-1)
        scores.extend(torch.sigmoid(logits).cpu().tolist())
        labels.extend(y.tolist())
        gens.extend(g)
    return binary_metrics(labels, scores), per_generator_metrics(labels, scores, gens)


def train_temporal(cfg: dict, args) -> dict:
    set_seed(cfg["train"]["seed"])
    device = get_device()
    cache_dir = resolve_path(cfg, "embed", "cache_dir")
    items = load_cached(cache_dir)
    if not items:
        raise SystemExit(f"No cached embeddings in {cache_dir}. Run 'embed' first.")
    n_real = sum(1 for i in items if i["label"] == 0)
    print(f"[temporal] {len(items)} videos (real={n_real}, fake={len(items) - n_real})")

    import random

    rng = random.Random(cfg["train"]["seed"])
    # stratified video-level split so val always has both classes
    by_label: dict[int, list] = {0: [], 1: []}
    for it in items:
        by_label[it["label"]].append(it)
    val_items: list = []
    train_items: list = []
    for lab, group in by_label.items():
        rng.shuffle(group)
        n_val = max(1, int(len(group) * 0.15))
        val_items.extend(group[:n_val])
        train_items.extend(group[n_val:])
    rng.shuffle(train_items)
    rng.shuffle(val_items)

    in_dim = train_items[0]["emb"].shape[1]
    print(f"[temporal] train={len(train_items)} val={len(val_items)} in_dim={in_dim}")

    train_loader = DataLoader(EmbDataset(train_items), batch_size=cfg["train"]["batch_size"],
                              shuffle=True, collate_fn=collate_emb)
    val_loader = DataLoader(EmbDataset(val_items), batch_size=cfg["train"]["batch_size"],
                            shuffle=False, collate_fn=collate_emb)

    model = TemporalTransformer(
        in_dim=in_dim,
        dim=cfg["model"]["dim"],
        depth=cfg["model"]["depth"],
        heads=cfg["model"]["heads"],
        mlp_ratio=cfg["model"]["mlp_ratio"],
        dropout=cfg["model"]["dropout"],
        pool=cfg["model"]["pool"],
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    history = []
    ckpt_path = resolve_path(cfg, "train", "ckpt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        running = 0.0
        t0 = time.time()
        for x, y, mask, _g, _v in train_loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device, dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            logits = model(x, mask).squeeze(-1)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            running += loss.item()
        overall, per_gen = evaluate_temporal(model, val_loader, device)
        auc = overall["auc"] if overall["auc"] == overall["auc"] else 0.0  # NaN guard
        print(f"[temporal] epoch {epoch} loss={running / max(len(train_loader),1):.4f} "
              f"val_auc={auc:.4f} acc={overall['acc']:.4f} ({time.time() - t0:.1f}s)")
        history.append({"epoch": epoch, **{f"val_{k}": v for k, v in overall.items()}})
        if auc > best_auc:
            best_auc = auc
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": {**cfg["model"], "in_dim": in_dim},
                    "val_auc": best_auc,
                    "epoch": epoch,
                },
                ckpt_path,
            )
            print(f"[temporal] saved best -> {ckpt_path} (auc={best_auc:.4f})")

    result = {"best_val_auc": best_auc, "history": history, "ckpt": str(ckpt_path), "per_generator_last": per_gen}
    (ckpt_path.parent / "temporal_results.json").write_text(json.dumps(result, indent=2))
    print(f"[temporal] done. best_val_auc={best_auc:.4f}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["embed", "train"])
    ap.add_argument("--config", default="configs/temporal.yaml")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.stage == "embed":
        embed_all(cfg, args)
    else:
        train_temporal(cfg, args)


if __name__ == "__main__":
    main()
