"""Prepare the wish096 FF++/CelebDF frame dataset for training.

Input (data/raw_frames):
    frame_dataset_v3/frame_dataset_v3/{fake,real}/*.jpg
    train_labels.csv / val_labels.csv / test_labels.csv
        columns: filepath, source, label, video, frame, split, det_box, det_prob

Output (data/processed):
    data/processed/{train,val,test}/{real,fake}/<video>_f<frame>.jpg

Uses hard links where possible (no extra disk), falling back to copy.
Also writes data/processed/manifest.json with per-split stats.

Usage:
    python -m deepfake_detection.prepare_data
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def run(args) -> None:
    raw = PROJECT_ROOT / "data" / "raw_frames"
    frames_root = raw / "frame_dataset_v3" / "frame_dataset_v3"
    out_root = PROJECT_ROOT / "data" / "processed"

    # Load all splits first so we can enforce video-level disjointness.
    # The Kaggle mirror's provided splits LEAK videos across train/val/test
    # (124 videos in train∩val, 78 in train∩test), which inflates metrics.
    # We treat val and test videos as sacred and drop them from train.
    frames = {}
    for split in ["train", "val", "test"]:
        frames[split] = pd.read_csv(raw / f"{split}_labels.csv")

    def vid_key(row) -> str:
        return f"{row['source']}_{row['video']}"

    val_vids = {vid_key(r) for _, r in frames["val"].iterrows()}
    test_vids = {vid_key(r) for _, r in frames["test"].iterrows()}
    held_out = val_vids | test_vids

    before = len(frames["train"])
    frames["train"] = frames["train"][
        ~frames["train"].apply(lambda r: vid_key(r) in held_out, axis=1)
    ]
    print(f"[prepare] removed {before - len(frames['train'])} leaked train rows "
          f"({len(held_out)} held-out videos)")

    manifest = {}
    total = 0
    for split in ["train", "val", "test"]:
        df = frames[split]
        for _, row in df.iterrows():
            fname = Path(str(row["filepath"])).name
            src = frames_root / ("fake" if row["label"] == 1 else "real") / fname
            if not src.exists():
                # try the other dir (safety)
                alt = frames_root / ("real" if row["label"] == 1 else "fake") / fname
                if alt.exists():
                    src = alt
                else:
                    continue
            cls = "fake" if row["label"] == 1 else "real"
            dst_dir = out_root / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            # prefix with source (ffpp_real/ffpp_fake/celeb_real/celeb_fake)
            # so generator/source can be recovered from the filename
            dst = dst_dir / f"{row['source']}_{row['video']}_f{row['frame']}.jpg"
            link_or_copy(src, dst)
            total += 1
        counts = df.groupby("label").size().to_dict()
        manifest[split] = {
            "rows": len(df),
            "real": int(counts.get(0, 0)),
            "fake": int(counts.get(1, 0)),
            "sources": df["source"].value_counts().to_dict(),
            "videos": int(df["video"].nunique()),
        }
        print(f"[prepare] {split}: {manifest[split]}")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[prepare] linked {total} frames -> {out_root}")


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    run(None)


if __name__ == "__main__":
    main()
