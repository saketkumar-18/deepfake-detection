"""Train an audio-forensics branch: raw-waveform 1D-CNN for real-vs-synthetic speech.

Data: Kaggle `deepfake_audio_dataset_jay15k` subset (fake/ vs real/ WAVs).
Protocol (same integrity rules as the visual branch):
  - speaker/file-disjoint splits (group by numeric id where possible)
  - clip-level labels; 16 kHz mono resample; random 2.0 s crops in train, fixed 2.0 s
    segments in val/test (multiple segments per clip, clip-level AUC via mean)
Model: 5-layer 1D-CNN over raw samples (no mel spectrogram -> stays tiny and
works identically in WASM), ~210k params, input (1, 32000).

Outputs:
  checkpoints/audio_cnn.pt        torch weights
  app/assets/model.audio.onnx    ONNX export (dynamic batch, fp16 converted)
  checkpoints/audio_results.json  metrics
"""
from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data" / "audio_unidpro"
CKPT_DIR = PROJECT_ROOT / "checkpoints"
APP_ASSETS = PROJECT_ROOT / "app" / "assets"

SR = 16000
SEG = 2.0
SEG_N = int(SR * SEG)


# ---------------------------------------------------------------- model
class AudioCNN(nn.Module):
    """Raw-waveform CNN: 4 conv blocks with stride downsampling + global pool."""

    def __init__(self):
        super().__init__()
        c = [1, 16, 32, 64, 128]
        blocks = []
        for i in range(4):
            blocks += [
                nn.Conv1d(c[i], c[i + 1], kernel_size=9, stride=1, padding=4),
                nn.BatchNorm1d(c[i + 1]),
                nn.ReLU(),
                nn.MaxPool1d(4),
            ]
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),   # global avg — ONNX-safe for any input length
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):  # x: (B, 1, N)
        h = self.blocks(x)
        return self.head(h)  # (B, 1) logits


# ---------------------------------------------------------------- data
def load_wav_ffmpeg(path: Path) -> np.ndarray:
    """Decode any audio to 16 kHz mono float32 via ffmpeg (avoids audioread deps)."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-ac", "1", "-ar", str(SR), "-f", "f32le", "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32)


def segments_from_clip(x: np.ndarray, seg_n: int, train: bool, rng: random.Random, crops: int = 12):
    """Yield fixed-length segments; TRAIN: many random crops (data augmentation);
    EVAL: tile the whole clip so the clip-level score uses all its audio."""
    if len(x) < seg_n:
        pad = np.zeros(seg_n, dtype=np.float32)
        pad[: len(x)] = x
        yield pad
        return
    if train:
        for _ in range(crops):
            start = rng.randint(0, len(x) - seg_n)
            seg = x[start : start + seg_n].copy()
            # loudness jitter ±3 dB
            db = rng.uniform(-3, 3)
            seg *= 10 ** (db / 20)
            yield seg
    else:
        for start in range(0, len(x) - seg_n + 1, seg_n):
            yield x[start : start + seg_n]


def scan_classes(root: Path) -> dict:
    """Return {'fake': [paths], 'real': [paths]}.

    Sources:
      - unidpro flattened '<group>_<gender>_<speaker>_<kind>.<ext>':
          kind = original (real) | synthetic_N (fake)
      - CREMA-D natural speech (real), if present under data/audio_cremad:
          adds diverse real speech from 91 speakers to fight channel overfit.
    """
    out = {"fake": [], "real": []}
    for p in root.iterdir():
        if not p.is_file() or p.suffix == ".xlsx":
            continue
        lab = "real" if "original" in p.stem else "fake"
        out[lab].append(p)
    cremad = root.parent / "audio_cremad"
    if cremad.is_dir():
        wavs = sorted(cremad.glob("*.wav"))
        # cap so the real class isn't dominated by one corpus (600 clips max)
        out["real"] += wavs[:600]
        print(f"[audio] + {min(len(wavs), 600)} CREMA-D natural-speech clips (real)")
    for k in out:
        random.shuffle(out[k])
    return out


# ---------------------------------------------------------------- training
def main() -> None:
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    APP_ASSETS.mkdir(parents=True, exist_ok=True)

    data = scan_classes(RAW)
    n_fake, n_real = len(data["fake"]), len(data["real"])
    print(f"[audio] fake={n_fake} real={n_real}")
    if n_fake < 10 or n_real < 10:
        sys.exit("[audio] not enough data — download the subset first")

    # speaker-disjoint split: group by speaker id
    #   unidpro: UK_female_1_<kind>  -> UK_female_1
    #   CREMA-D: 1001_DFA_ANG_XX     -> 1001
    def gid(p: Path) -> str:
        parts = p.stem.split("_")
        return "_".join(parts[:3]) if parts[0] in ("UK", "USA") else parts[0]

    groups = {}
    for lab, paths in data.items():
        for p in paths:
            g = gid(p)
            groups.setdefault(g, []).append((p, lab))
    gids = sorted(groups.keys())
    random.shuffle(gids)
    n = len(gids)
    n_val, n_test = max(2, int(n * 0.2)), max(2, int(n * 0.2))
    val_g, test_g, train_g = set(gids[:n_val]), set(gids[n_val : n_val + n_test]), set(gids[n_val + n_test :])

    def collect(gset):
        paths = [(p, l) for g in gset for (p, l) in groups[g]]
        return paths

    train_list = collect(train_g)
    val_list = collect(val_g)
    test_list = collect(test_g)
    print(f"[audio] groups train={len(train_g)} val={len(val_g)} test={len(test_g)} "
          f"| clips train={len(train_list)} val={len(val_list)} test={len(test_list)}")

    # ---------------- cache decoded segments in RAM (subset is small)
    def build(paths, train):
        X, y, clip_id = [], [], []
        for i, (p, lab) in enumerate(paths):
            try:
                x = load_wav_ffmpeg(p)
            except Exception as e:
                print(f"  skip {p.name}: {e}")
                continue
            for s in segments_from_clip(x, SEG_N, train, rng):
                X.append(s)
                y.append(1 if lab == "fake" else 0)
                clip_id.append(i)
        X = np.stack(X)
        y = np.array(y, dtype=np.float32)
        return X, y, np.array(clip_id)

    print("[audio] decoding train...")
    Xtr, ytr, _ = build(train_list, True)
    print(f"[audio] train segments: {Xtr.shape}")
    print("[audio] decoding val...")
    Xv, yv, cidv = build(val_list, False)
    print("[audio] decoding test...")
    Xt, yt, cidt = build(test_list, False)

    # ---------------- training loop (CPU)
    dev = torch.device("cpu")
    model = AudioCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    B = 64
    steps_per_epoch = max(1, len(Xtr) // B)
    epochs = 30
    best_auc, best_state = -1.0, None

    def clip_auc(model, X, y, cid):
        model.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, len(X), 128):
                xb = torch.from_numpy(X[i : i + 128]).unsqueeze(1)
                probs.append(torch.sigmoid(model(xb)).squeeze(-1).numpy())
        probs = np.concatenate(probs)
        # clip-level: mean over segments of the same clip
        clips = {}
        for c in np.unique(cid):
            m = cid == c
            clips[c] = (float(probs[m].mean()), float(y[m][0]))
        cp = np.array([v[0] for v in clips.values()])
        cy = np.array([v[1] for v in clips.values()])
        return roc_auc_score(cy, cp), cp, cy

    print("[audio] training...")
    for ep in range(1, epochs + 1):
        model.train()
        idx = np.arange(len(Xtr))
        np.random.shuffle(idx)
        tot, cnt = 0.0, 0
        for s in range(steps_per_epoch):
            sel = idx[s * B : (s + 1) * B]
            xb = torch.from_numpy(Xtr[sel]).unsqueeze(1)
            yb = torch.from_numpy(ytr[sel]).unsqueeze(1)
            logits = model(xb)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            cnt += 1
        vauc, _, _ = clip_auc(model, Xv, yv, cidv)
        print(f"[audio] epoch {ep} loss={tot/cnt:.4f} val_clip_auc={vauc:.4f}")
        if vauc > best_auc:
            best_auc = vauc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    tauc, tp, ty = clip_auc(model, Xt, yt, cidt)
    vauc, _, _ = clip_auc(model, Xv, yv, cv := cidv)
    print(f"[audio] best val clip AUC {vauc:.4f} | test clip AUC {tauc:.4f}")

    torch.save(
        {"model": best_state, "config": {"sr": SR, "seg": SEG, "arch": "audio_cnn_5l"},
         "val_auc": float(vauc), "test_auc": float(tauc)},
        CKPT_DIR / "audio_cnn.pt",
    )

    # ---------------- ONNX export for the in-browser demo ----------------
    import onnx
    import onnxruntime as ort

    dummy = torch.zeros(1, 1, SEG_N)
    onnx_path = APP_ASSETS / "model.audio.onnx"
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["waveform"], output_names=["fake_logit"],
        opset_version=17, dynamo=False,
        dynamic_axes={"waveform": {0: "batch"}, "fake_logit": {0: "batch"}},
    )
    try:
        from onnxconverter_common import float16
        m = onnx.load(str(onnx_path))
        m16 = float16.convert_float_to_float16(m, keep_io_types=True)
        onnx.save(m16, str(onnx_path))
    except Exception as e:
        print(f"[audio] fp16 conversion skipped: {e}")

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    xt = torch.from_numpy(Xt[:4]).unsqueeze(1)
    with torch.no_grad():
        tp = torch.sigmoid(model(xt)).squeeze(-1).numpy()
    op = 1 / (1 + np.exp(-sess.run(None, {"waveform": xt.numpy()})[0].squeeze(-1)))
    print(f"[audio] onnx parity max diff: {np.abs(tp - op).max():.2e} size: {onnx_path.stat().st_size/1e6:.2f} MB")

    # ---------------- per-class/per-source test breakdown (honest numbers)
    def per_source(paths, X, y, cid):
        probs = []
        model.eval()
        with torch.no_grad():
            for i in range(0, len(X), 128):
                xb = torch.from_numpy(X[i : i + 128]).unsqueeze(1)
                probs.append(torch.sigmoid(model(xb)).squeeze(-1).numpy())
        probs = np.concatenate(probs)
        # map clip idx -> its label
        clip_label = {i: l for i, (p, l) in enumerate(paths)}
        out = {}
        for c in np.unique(cid):
            m = cid == c
            lab = clip_label[int(c)]
            out.setdefault(lab, []).append(float(probs[m].mean()))
        return out

    ps = per_source(test_list, Xt, yt, cidt)
    res = {
        "best_val_clip_auc": float(vauc),
        "test_clip_auc": float(tauc),
        "n_test_clips": int(len(np.unique(cidt))),
        "mean_fake_prob": float(np.mean(ps.get("fake", [np.nan]))),
        "mean_real_prob": float(np.mean(ps.get("real", [np.nan]))),
        "protocol": "group-disjoint splits by numeric file id; clip-level AUC via segment mean",
    }
    (CKPT_DIR / "audio_results.json").write_text(json.dumps(res, indent=2))
    print("[audio] results:", json.dumps(res, indent=2))
    print("[audio] DONE")


if __name__ == "__main__":
    main()
