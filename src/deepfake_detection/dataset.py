"""Frame-level and video-level datasets for deepfake detection.

Supports two layouts:
1. Pre-extracted frame folders:  <root>/{real,fake}/<video_id>_f<frame>.jpg
   or <root>/<generator>/<video_id>_f<frame>.jpg (FF++ style, generator = class).
2. Processed Celeb-DF crops:     <root>/{Celeb-real,Celeb-synthesis}/<vid>/frame_%04d.jpg
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# filename like celeb_fake_id0_id16_0000_f468.jpg  or  000_003_f12.jpg
_FRAME_RE = re.compile(r"^(?P<vid>.+)_f(?P<fidx>\d+)\.(?:jpg|jpeg|png)$", re.I)

GENERATOR_LABELS = [
    "Deepfakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
    "FaceShifter",
    "DeepFakeDetection",
    "CelebDF",
]


def parse_frame_name(name: str) -> tuple[str, int] | None:
    """Return (video_id, frame_index) parsed from a frame filename."""
    m = _FRAME_RE.match(name)
    if not m:
        return None
    return m.group("vid"), int(m.group("fidx"))


def build_train_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)),
        ]
    )


def build_eval_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@dataclass
class FrameSample:
    path: Path
    label: int          # 0 real, 1 fake
    video_id: str
    generator: str      # 'real' or generator name
    frame_idx: int = 0


def load_split(root: Path, split: str) -> list[FrameSample]:
    """Load one official split from <root>/<split>/{real,fake,...} layout.

    Falls back to scanning the whole root if the split dir doesn't exist.
    """
    root = Path(root)
    split_dir = root / split
    if split_dir.is_dir():
        return scan_frame_tree(split_dir)
    return scan_frame_tree(root)


def scan_frame_tree(
    root: Path,
    class_map: dict[str, int] | None = None,
) -> list[FrameSample]:
    """Scan a directory tree of pre-extracted frames.

    Expected layouts (auto-detected per top-level dir):
      root/real/...  root/fake/...                      -> binary
      root/<Generator>/...  (FF++ generator names)      -> fake, generator tagged
      root/Celeb-real/... root/Celeb-synthesis/...      -> binary (CelebDF)
    """
    if class_map is None:
        class_map = {"real": 0, "fake": 1}
    samples: list[FrameSample] = []
    root = Path(root)
    for top in sorted(p for p in root.iterdir() if p.is_dir()):
        tname = top.name.lower()
        if tname in ("real", "celeb-real", "original"):
            label, gen = 0, "real"
        elif tname in ("fake", "celeb-synthesis"):
            label, gen = 1, "CelebDF" if "celeb" in tname else "fake"
        elif top.name in GENERATOR_LABELS:
            label, gen = 1, top.name
        else:
            continue
        for img in top.rglob("*"):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            parsed = parse_frame_name(img.name)
            vid, fidx = parsed if parsed else (img.stem, 0)
            # recover source/generator tag from filename prefix when generic
            if gen == "fake":
                if vid.startswith("celeb_fake"):
                    gen = "CelebDF"
                elif vid.startswith("ffpp_fake"):
                    gen = "FF++"
            samples.append(FrameSample(img, label, vid, gen, fidx))
    return samples


class FrameDataset(Dataset):
    """Frame-level binary dataset."""

    def __init__(
        self,
        samples: list[FrameSample],
        transform=None,
        max_per_video: int | None = None,
        seed: int = 42,
    ):
        self.transform = transform
        if max_per_video is not None:
            rng = random.Random(seed)
            by_vid: dict[str, list[FrameSample]] = {}
            for s in samples:
                by_vid.setdefault(s.video_id, []).append(s)
            kept: list[FrameSample] = []
            for vid, group in by_vid.items():
                group.sort(key=lambda s: s.frame_idx)
                if len(group) > max_per_video:
                    step = len(group) / max_per_video
                    idxs = sorted({int(i * step) for i in range(max_per_video)})
                    group = [group[i] for i in idxs]
                kept.extend(group)
            samples = kept
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        img = Image.open(s.path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, s.label, s.generator, s.video_id


@dataclass
class VideoClip:
    video_id: str
    label: int
    generator: str
    frames: list[Path] = field(default_factory=list)
    split: str = "all"


def group_videos(samples: list[FrameSample], frames_per_video: int = 16) -> list[VideoClip]:
    """Group frames into video clips, evenly subsampling frames_per_video frames."""
    by_vid: dict[tuple[str, int], list[FrameSample]] = {}
    for s in samples:
        by_vid.setdefault((s.video_id, s.label), []).append(s)
    clips: list[VideoClip] = []
    for (vid, label), group in by_vid.items():
        group.sort(key=lambda s: s.frame_idx)
        if len(group) > frames_per_video:
            step = len(group) / frames_per_video
            idxs = sorted({int(i * step) for i in range(frames_per_video)})
            group = [group[i] for i in idxs]
        clips.append(
            VideoClip(
                video_id=vid,
                label=label,
                generator=group[0].generator,
                frames=[s.path for s in group],
            )
        )
    return clips


class VideoDataset(Dataset):
    """Video-level dataset: returns (T, C, H, W) tensor per clip."""

    def __init__(self, clips: list[VideoClip], transform=None):
        self.clips = clips
        self.transform = transform

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, i: int):
        clip = self.clips[i]
        tensors = []
        for p in clip.frames:
            img = Image.open(p).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            tensors.append(img)
        x = torch.stack(tensors)  # (T, C, H, W)
        return x, clip.label, clip.generator, clip.video_id


def collate_video(batch):
    """Collate variable-length clips by padding to max T in batch."""
    xs, labels, gens, vids = zip(*batch)
    max_t = max(x.shape[0] for x in xs)
    padded = []
    masks = []
    for x in xs:
        t = x.shape[0]
        pad = torch.zeros(max_t - t, *x.shape[1:], dtype=x.dtype)
        padded.append(torch.cat([x, pad], dim=0))
        m = torch.zeros(max_t, dtype=torch.bool)
        m[:t] = True
        masks.append(m)
    return (
        torch.stack(padded),
        torch.tensor(labels, dtype=torch.long),
        torch.stack(masks),
        list(gens),
        list(vids),
    )
