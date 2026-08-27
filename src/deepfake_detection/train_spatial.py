"""Train the spatial (frame-level) detector.

Usage:
    python -m deepfake_detection.train_spatial --config configs/spatial.yaml \
        [--data-root data/processed] [--epochs 3] [--limit N]

Designed for CPU: small backbone, mixed-precision off, modest batch.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import load_config, resolve_path
from .dataset import (
    FrameDataset,
    build_eval_transform,
    build_train_transform,
    scan_frame_tree,
)
from .metrics import binary_metrics, per_generator_metrics
from .models.spatial import SpatialDetector
from .utils import get_device, set_seed


def make_sampler(dataset: FrameDataset) -> WeightedRandomSampler:
    """Balance real/fake by inverse-frequency sampling."""
    labels = [s.label for s in dataset.samples]
    counts = [labels.count(0), labels.count(1)]
    weights = [1.0 / max(counts[l], 1) for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_scores, all_labels, all_gens = [], [], []
    for imgs, labels, gens, _vids in loader:
        imgs = imgs.to(device)
        logits = model(imgs).squeeze(-1)
        probs = torch.sigmoid(logits)
        all_scores.extend(probs.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_gens.extend(gens)
    overall = binary_metrics(all_labels, all_scores)
    per_gen = per_generator_metrics(all_labels, all_scores, all_gens)
    return overall, per_gen


def train(cfg: dict, args) -> dict:
    set_seed(cfg["train"]["seed"], cfg["train"].get("num_threads"))
    device = get_device()
    print(f"[spatial] device={device}")

    data_root = Path(args.data_root) if args.data_root else resolve_path(cfg, "data", "processed_root")
    img_size = cfg["data"]["img_size"]

    print(f"[spatial] scanning frames under {data_root} ...")
    from .dataset import load_split

    train_samples = load_split(data_root, "train")
    val_samples = load_split(data_root, "val")
    if not train_samples:
        # no official splits: fall back to scanning everything + random split
        samples = scan_frame_tree(data_root)
        import random

        rng = random.Random(cfg["train"]["seed"])
        vids_by_label: dict[int, list[str]] = {0: [], 1: []}
        vid_label = {}
        for s in samples:
            if s.video_id not in vid_label:
                vid_label[s.video_id] = s.label
                vids_by_label[s.label].append(s.video_id)
        val_vids = set()
        for lab, vids in vids_by_label.items():
            rng.shuffle(vids)
            n_v = max(1, int(len(vids) * 0.15))
            val_vids.update(vids[:n_v])
        train_samples = [s for s in samples if s.video_id not in val_vids]
        val_samples = [s for s in samples if s.video_id in val_vids]
    n_real = sum(1 for s in train_samples if s.label == 0)
    n_fake = sum(1 for s in train_samples if s.label == 1)
    print(f"[spatial] train frames: {len(train_samples)} (real={n_real}, fake={n_fake}) "
          f"val frames: {len(val_samples)}")
    if not train_samples:
        raise SystemExit(f"No frames found under {data_root}. Run preprocessing first.")

    if args.limit:
        train_samples = train_samples[: args.limit]
        val_samples = val_samples[: max(64, args.limit // 5)]

    train_vids = len({s.video_id for s in train_samples})
    val_vids = len({s.video_id for s in val_samples})
    print(f"[spatial] train frames={len(train_samples)} val frames={len(val_samples)} "
          f"(train vids={train_vids}, val vids={val_vids})")

    train_ds = FrameDataset(train_samples, build_train_transform(img_size))
    val_ds = FrameDataset(val_samples, build_eval_transform(img_size))

    sampler = make_sampler(train_ds)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["data"]["batch_size"], sampler=sampler,
        num_workers=cfg["data"]["workers"], pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["eval"]["batch_size"], shuffle=False,
        num_workers=cfg["data"]["workers"],
    )

    model = SpatialDetector(
        backbone=cfg["model"]["backbone"],
        pretrained=cfg["model"]["pretrained"],
        num_classes=cfg["model"]["num_classes"],
        drop_rate=cfg["model"]["drop_rate"],
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()

    epochs = args.epochs if args.epochs else cfg["train"]["epochs_finetune"]
    best_auc = 0.0
    history = []
    ckpt_path = resolve_path(cfg, "train", "ckpt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # phase 1: head only (backbone frozen, no grads); phase 2+: full fine-tune
        if epoch == 1:
            for p in model.net.parameters():
                p.requires_grad = False
            params = model.param_groups(0.0, cfg["train"]["lr_head"], cfg["train"]["weight_decay"])
        else:
            for p in model.net.parameters():
                p.requires_grad = True
            params = model.param_groups(cfg["train"]["lr_finetune"], cfg["train"]["lr_finetune"], cfg["train"]["weight_decay"])
        opt = torch.optim.AdamW(params)

        model.train()
        t0 = time.time()
        running = 0.0
        for step, (imgs, labels, _g, _v) in enumerate(train_loader):
            imgs = imgs.to(device)
            y = labels.to(device, dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            logits = model(imgs).squeeze(-1)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            running += loss.item()
            if step % 20 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss={loss.item():.4f}")
        train_loss = running / max(len(train_loader), 1)

        overall, per_gen = evaluate(model, val_loader, device)
        dt = time.time() - t0
        auc = overall["auc"] if overall["auc"] == overall["auc"] else 0.0  # NaN guard
        print(f"[spatial] epoch {epoch} loss={train_loss:.4f} val_auc={auc:.4f} "
              f"val_acc={overall['acc']:.4f} ({dt:.1f}s)")
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in overall.items()}})

        if auc > best_auc:
            best_auc = auc
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": {
                        "backbone": cfg["model"]["backbone"],
                        "num_classes": cfg["model"]["num_classes"],
                        "drop_rate": cfg["model"]["drop_rate"],
                        "img_size": img_size,
                    },
                    "val_auc": best_auc,
                    "epoch": epoch,
                },
                ckpt_path,
            )
            print(f"[spatial] saved best ckpt -> {ckpt_path} (auc={best_auc:.4f})")

    result = {"best_val_auc": best_auc, "history": history, "ckpt": str(ckpt_path), "per_generator_last": per_gen}
    out = ckpt_path.parent / "spatial_results.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"[spatial] done. best_val_auc={best_auc:.4f}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/spatial.yaml")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap train frames (smoke test)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    train(cfg, args)


if __name__ == "__main__":
    main()
