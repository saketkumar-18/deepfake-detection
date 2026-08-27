"""Artifact analysis: which manipulation traces generalize across generators?

Two complementary analyses:

1. SPECTRAL (FFT): GAN/synthesis pipelines leave characteristic fingerprints
   in the frequency domain (upsampling grids, decoder transposes). We compute
   the azimuthally-averaged radial power spectrum for real vs. each fake
   generator, and report the log-spectral divergence (L1 between normalized
   radial profiles). Generators whose spectra deviate from real in the SAME
   mid/high-frequency bands share an artifact family -> likely to transfer.

2. SPATIAL ATTENTION (Grad-CAM): where does the detector look for each
   generator? We aggregate Grad-CAM heatmaps per generator and report the
   fraction of attention mass on canonical face regions (eyes/mouth/cheeks
   vs. boundary/hairline), using a simple radial-zone heuristic. Consistent
   attention placement across generators => the model relies on a
   generalizable cue rather than dataset shortcuts.

Outputs JSON + matplotlib figures into reports/.

Usage:
    python -m deepfake_detection.analyze_artifacts --data-root data/processed \
        [--ckpt checkpoints/spatial_effb0.pt] [--max-images 400]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .config import PROJECT_ROOT


# ---------------------------------------------------------------- spectral
def radial_power_spectrum(img: np.ndarray, size: int = 128) -> np.ndarray:
    """Azimuthally averaged log power spectrum of a grayscale image crop."""
    import cv2

    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    g = (g - g.mean()) / (g.std() + 1e-6)
    f = np.fft.fft2(g)
    fshift = np.fft.fftshift(f)
    power = np.log1p(np.abs(fshift) ** 2)
    cy = cx = size // 2
    y, x = np.ogrid[:size, :size]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    max_r = min(cy, cx)
    radial = np.zeros(max_r)
    for ri in range(max_r):
        mask = r == ri
        if mask.any():
            radial[ri] = power[mask].mean()
    return radial


def spectral_profiles(data_root: Path, max_images: int, img_size: int = 128):
    """Compute mean radial spectra per class/generator."""
    from .dataset import scan_frame_tree

    samples = scan_frame_tree(data_root)
    by_gen: dict[str, list] = defaultdict(list)
    for s in samples:
        by_gen[s.generator].append(s)

    profiles = {}
    for gen, ss in by_gen.items():
        acc = []
        for s in ss[:max_images]:
            try:
                img = np.asarray(Image.open(s.path).convert("RGB"))
                acc.append(radial_power_spectrum(img, img_size))
            except Exception:  # noqa: BLE001
                continue
        if acc:
            profiles[gen] = np.mean(acc, axis=0)
            print(f"  spectral: {gen:20s} n={len(acc)}")
    return profiles


def spectral_divergence(profiles: dict) -> dict:
    """L1 divergence of each generator's normalized spectrum vs real."""
    real = profiles.get("real")
    if real is None:
        return {}
    rn = real / (real.sum() + 1e-9)
    out = {}
    for gen, prof in profiles.items():
        if gen == "real":
            continue
        gn = prof / (prof.sum() + 1e-9)
        diff = gn - rn
        out[gen] = {
            "l1": float(np.abs(diff).sum()),
            # which frequency bands deviate most (low/mid/high thirds)
            "band_deviation": {
                "low": float(np.abs(diff[: len(diff) // 3]).sum()),
                "mid": float(np.abs(diff[len(diff) // 3 : 2 * len(diff) // 3]).sum()),
                "high": float(np.abs(diff[2 * len(diff) // 3 :]).sum()),
            },
        }
    return out


# ---------------------------------------------------------------- grad-cam
class GradCAM:
    """Minimal Grad-CAM for the last conv block of a timm CNN."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.acts = None
        self.grads = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, module, inp, out):
        self.acts = out.detach()

    def _bwd(self, module, grad_in, grad_out):
        self.grads = grad_out[0].detach()

    def __call__(self, x):
        import torch

        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        self.model.zero_grad()
        logits.backward(torch.ones_like(logits))
        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.squeeze().cpu().numpy()


def attention_zones(data_root: Path, ckpt: str, max_images: int, img_size: int = 224):
    """Aggregate Grad-CAM mass into center vs. boundary zones per generator.

    Heuristic: face manipulations concentrate in the central face region;
    boundary-heavy attention suggests the model uses compression/crop
    shortcuts. We report center_mass fraction per generator.
    """
    import cv2
    import torch

    from .dataset import build_eval_transform, scan_frame_tree
    from .models.spatial import load_spatial

    device = torch.device("cpu")
    model = load_spatial(ckpt).to(device).eval()
    # find last conv block: timm efficientnet -> model.net.conv_head if present
    target = None
    for name in ["conv_head", "bn2", "blocks"]:
        if hasattr(model.net, name):
            target = getattr(model.net, name)
            break
    if target is None:
        print("  [gradcam] no suitable target layer; skipping")
        return {}
    cam = GradCAM(model, target)
    transform = build_eval_transform(img_size)

    samples = scan_frame_tree(data_root)
    by_gen: dict[str, list] = defaultdict(list)
    for s in samples:
        by_gen[s.generator].append(s)

    results = {}
    for gen, ss in by_gen.items():
        center_masses = []
        for s in ss[:max_images]:
            try:
                img = Image.open(s.path).convert("RGB")
                x = transform(img).unsqueeze(0).to(device)
                heat = cam(x)
                heat = cv2.resize(heat, (img_size, img_size))
                h, w = heat.shape
                cy0, cy1 = h // 4, 3 * h // 4
                cx0, cx1 = w // 4, 3 * w // 4
                total = heat.sum() + 1e-9
                center = heat[cy0:cy1, cx0:cx1].sum()
                center_masses.append(float(center / total))
            except Exception:  # noqa: BLE001
                continue
        if center_masses:
            results[gen] = {
                "center_mass_mean": float(np.mean(center_masses)),
                "center_mass_std": float(np.std(center_masses)),
                "n": len(center_masses),
            }
            print(f"  gradcam: {gen:20s} center_mass={results[gen]['center_mass_mean']:.3f} n={len(center_masses)}")
    return results


# ---------------------------------------------------------------- plots
def plot_spectra(profiles: dict, out_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for gen, prof in profiles.items():
        n = prof / (prof.sum() + 1e-9)
        ax.plot(n, label=gen, lw=1.5)
    ax.set_xlabel("radial frequency (0=DC)")
    ax.set_ylabel("normalized power")
    ax.set_title("Radial power spectra: real vs. generators")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  plot -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/processed")
    ap.add_argument("--ckpt", default="checkpoints/spatial_effb0.pt")
    ap.add_argument("--max-images", type=int, default=400)
    ap.add_argument("--skip-gradcam", action="store_true")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(exist_ok=True)

    print("[artifacts] computing spectral profiles ...")
    profiles = spectral_profiles(data_root, args.max_images)
    divergence = spectral_divergence(profiles)
    plot_spectra(profiles, reports / "spectral_profiles.png")

    gradcam = {}
    ckpt_path = PROJECT_ROOT / args.ckpt
    if not args.skip_gradcam and ckpt_path.exists():
        print("[artifacts] computing Grad-CAM attention zones ...")
        gradcam = attention_zones(data_root, str(ckpt_path), min(args.max_images, 150))

    result = {"spectral_divergence": divergence, "gradcam_center_mass": gradcam}
    out = reports / "artifacts.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"[artifacts] saved -> {out}")

    # readable summary
    print("\n=== Spectral divergence vs real (higher = stronger artifact) ===")
    for gen, d in sorted(divergence.items(), key=lambda kv: -kv[1]["l1"]):
        b = d["band_deviation"]
        print(f"  {gen:20s} L1={d['l1']:.4f}  low={b['low']:.4f} mid={b['mid']:.4f} high={b['high']:.4f}")


if __name__ == "__main__":
    main()
