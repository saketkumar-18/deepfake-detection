"""Prepare per-generator FF++ frame layout using official FF++ splits.

Input (data/raw_ffgen):
    Frames(cropped+aligned)/{Original,Deepfakes,Face2Face,FaceSwap,
                             NeuralTextures,FaceShifter}/<pair>_f<idx>.jpg
    data/ffpp_{train,val,test}.json  (official video-pair splits)

Output (data/processed_generators):
    {train,val,test}/{real,Deepfakes,Face2Face,FaceSwap,NeuralTextures,FaceShifter}/...

Hard-links frames; split membership comes from the official pair lists so
there is zero identity leakage between splits.

Usage:
    python -m deepfake_detection.prepare_ffgen
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .config import PROJECT_ROOT

GEN_DIRS = {
    "Original": "real",
    "Deepfakes": "Deepfakes",
    "Face2Face": "Face2Face",
    "FaceSwap": "FaceSwap",
    "NeuralTextures": "NeuralTextures",
    "FaceShifter": "FaceShifter",
}


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def run() -> None:
    raw = PROJECT_ROOT / "data" / "raw_ffgen" / "Frames(cropped+aligned)"
    out_root = PROJECT_ROOT / "data" / "processed_generators"

    pair_split: dict[str, str] = {}
    single_split: dict[str, str] = {}
    for split in ["train", "val", "test"]:
        pairs = json.loads((PROJECT_ROOT / "data" / f"ffpp_{split}.json").read_text())
        for a, b in pairs:
            pair_split[f"{a}_{b}"] = split
            pair_split[f"{b}_{a}"] = split  # swapped order appears too
            single_split[a] = split          # Original videos use single ids
            single_split[b] = split

    stats: dict[str, dict[str, int]] = {}
    missing = 0
    for src_dir, cls in GEN_DIRS.items():
        d = raw / src_dir
        if not d.is_dir():
            print(f"[ffgen] missing dir {d}")
            continue
        for img in d.iterdir():
            if img.suffix.lower() != ".jpg":
                continue
            # name like 000_003_f0.jpg -> pair = 000_003
            stem = img.stem
            parts = stem.rsplit("_f", 1)
            pair = parts[0]
            split = pair_split.get(pair) or single_split.get(pair)
            if split is None:
                missing += 1
                continue
            dst_dir = out_root / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            link_or_copy(img, dst_dir / img.name)
            stats.setdefault(split, {})
            stats[split][cls] = stats[split].get(cls, 0) + 1

    for split, d in stats.items():
        print(f"[ffgen] {split}: {d}")
    if missing:
        print(f"[ffgen] WARNING: {missing} frames had no split assignment")
    (out_root / "manifest.json").write_text(json.dumps(stats, indent=2))
    print(f"[ffgen] done -> {out_root}")


if __name__ == "__main__":
    run()
