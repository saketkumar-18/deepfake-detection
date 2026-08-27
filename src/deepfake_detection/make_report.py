"""Assemble the final capstone report from all JSON artifacts in reports/.

Reads: spatial_results.json, temporal_results.json, generalization.json,
       artifacts.json, benchmark.json (+ per-generator eval if present)
Writes: reports/REPORT.md

Usage:
    python -m deepfake_detection.make_report
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import PROJECT_ROOT

REPORTS = PROJECT_ROOT / "reports"
CHECKPOINTS = PROJECT_ROOT / "checkpoints"


def load(name: str) -> dict | None:
    for root in (REPORTS, CHECKPOINTS):
        p = root / name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def fmt_auc(m: dict | None) -> str:
    if not m or "auc" not in m:
        return "—"
    return f"{m['auc']:.4f}"


def main() -> None:
    spatial = load("spatial_results.json")
    temporal = load("temporal_results.json")
    gen = load("generalization.json")
    gen_test = load("generalization_generators.json")
    artifacts = load("artifacts.json")
    bench = load("benchmark.json")

    L: list[str] = []
    L.append("# Deepfake Video Detection — Experiment Report\n")
    L.append("Spatial-temporal CNN/Transformer forensics on FaceForensics++ (5 generators) "
             "and Celeb-DF v2 (held-out cross-dataset). CPU training (16-core, no GPU).\n")
    L.append("> **Protocol integrity:** all numbers below use video-level-disjoint splits "
             "(the Kaggle mirror's shipped splits leaked 124 val + 78 test videos into train; "
             "1,044 leaked rows were removed) and the temporal model is length-controlled "
             "(T=2 frames per clip — the raw data has a frame-count shortcut: real clips carry "
             "12 frames, fakes 2–3, which alone yields a fake-perfect AUC of 1.0).\n")

    # ---- headline numbers
    L.append("## Headline results\n")
    L.append("| Model | Val AUC | Notes |")
    L.append("|---|---|---|")
    if spatial:
        L.append(f"| Spatial (EfficientNet-B0, frame-level) | {fmt_auc({'auc': spatial.get('best_val_auc')})} | "
                 f"epoch {spatial.get('history', [{}])[-1].get('epoch', '?')} |")
    if bench:
        for k in ["spatial_agg_mean", "spatial_agg_max", "spatial_agg_topk", "temporal_transformer"]:
            if k in bench:
                L.append(f"| Video-level: {k} | {fmt_auc(bench[k])} | |")
    L.append("")

    # ---- cross-generator
    L.append("## Cross-generator generalization (frame AUC)\n")
    L.append("Research question: *which artifacts generalize across generators?*\n")
    for title, data in [("Cross-source held-out test (Celeb-DF v2 vs FF++)", gen),
                        ("Per-generator FF++ test (cross-preprocessing)", gen_test)]:
        if not data:
            continue
        L.append(f"### {title}\n")
        L.append("| Generator / source | AUC | Acc | n |")
        L.append("|---|---|---|---|")
        for g, m in sorted(data.items()):
            if g == "overall":
                continue
            L.append(f"| {g} | {fmt_auc(m)} | {m.get('acc', float('nan')):.4f} | {m.get('n', 0)} |")
        if "overall" in data:
            L.append(f"| **OVERALL** | **{fmt_auc(data['overall'])}** | "
                     f"{data['overall'].get('acc', float('nan')):.4f} | {data['overall'].get('n', 0)} |")
        L.append("")

    # ---- artifacts
    if artifacts:
        L.append("## Artifact analysis\n")
        sd = artifacts.get("spectral_divergence", {})
        if sd:
            L.append("### Spectral fingerprints (radial FFT divergence vs real)\n")
            L.append("Higher L1 = stronger deviation from real. Band columns show WHERE "
                     "in the frequency spectrum the artifact lives.\n")
            L.append("| Generator | L1 | low-freq | mid-freq | high-freq |")
            L.append("|---|---|---|---|---|")
            for g, d in sorted(sd.items(), key=lambda kv: -kv[1]["l1"]):
                b = d["band_deviation"]
                L.append(f"| {g} | {d['l1']:.4f} | {b['low']:.4f} | {b['mid']:.4f} | {b['high']:.4f} |")
            L.append("")
        gc = artifacts.get("gradcam_center_mass", {})
        if gc:
            L.append("### Detector attention (Grad-CAM center mass)\n")
            L.append("Fraction of attention mass in the central face region. Similar values "
                     "across generators ⇒ the model uses a consistent, generalizable cue.\n")
            L.append("| Generator | center mass | n |")
            L.append("|---|---|---|")
            for g, d in sorted(gc.items()):
                L.append(f"| {g} | {d['center_mass_mean']:.3f} ± {d['center_mass_std']:.3f} | {d['n']} |")
            L.append("")

    # ---- training curves
    if spatial and spatial.get("history"):
        L.append("## Spatial training history\n")
        L.append("| epoch | train loss | val AUC | val acc |")
        L.append("|---|---|---|---|")
        for h in spatial["history"]:
            L.append(f"| {h['epoch']} | {h.get('train_loss', float('nan')):.4f} | "
                     f"{h.get('val_auc', float('nan')):.4f} | {h.get('val_acc', float('nan')):.4f} |")
        L.append("")

    out = REPORTS / "REPORT.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"[report] wrote {out}")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
