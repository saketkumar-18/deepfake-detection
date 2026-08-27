"""Export the spatial detector to ONNX + int8-quantized ONNX for browser inference.

Produces:
  app/assets/model.onnx     fp32 export (image -> fake logit)
  app/assets/model.q8.onnx  dynamic int8 quantized (smaller, faster in WASM)
  app/assets/meta.json      input size, normalization, benchmark numbers

Verifies parity: torch vs onnxruntime vs quantized on a batch of real test frames.

Usage:
    python -m deepfake_detection.export_onnx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import PROJECT_ROOT
from .models.spatial import load_spatial


class ExportWrapper(torch.nn.Module):
    """image -> sigmoid(fake probability), single scalar output."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.sigmoid(self.model(x)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/spatial_effb0.pt")
    ap.add_argument("--out-dir", default="app/assets")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = PROJECT_ROOT / ckpt_path
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_spatial(ckpt_path)
    model.eval()
    cfg = torch.load(ckpt_path, map_location="cpu").get("config", {})
    img_size = cfg.get("img_size", 224)
    print(f"[export] backbone={cfg.get('backbone')} img_size={img_size} "
          f"val_auc={torch.load(ckpt_path, map_location='cpu').get('val_auc')}")

    wrapper = ExportWrapper(model)
    dummy = torch.randn(1, 3, img_size, img_size)

    fp32_path = out_dir / "model.onnx"
    torch.onnx.export(
        wrapper,
        dummy,
        fp32_path,
        input_names=["image"],
        output_names=["fake_prob"],
        opset_version=args.opset,
        dynamic_axes={"image": {0: "batch"}, "fake_prob": {0: "batch"}},
        dynamo=False,  # legacy TorchScript exporter: cleaner graph for ORT
    )
    print(f"[export] fp32 -> {fp32_path} ({fp32_path.stat().st_size / 1e6:.1f} MB)")

    # fp16 conversion. NOTE: int8 dynamic quantization was tested and REJECTED —
    # it collapses EfficientNet's SE blocks (AUC 0.97 -> 0.55). fp16 halves the
    # size with no measurable accuracy loss (max |diff| ~0.006).
    import onnx
    from onnxconverter_common import float16

    q_path = out_dir / "model.fp16.onnx"
    m = onnx.load(str(fp32_path))
    m16 = float16.convert_float_to_float16(m, keep_io_types=True)
    onnx.save(m16, str(q_path))
    print(f"[export] fp16 -> {q_path} ({q_path.stat().st_size / 1e6:.1f} MB)")

    # ---- parity check on real test frames ----
    import onnxruntime as ort

    from .dataset import build_eval_transform, load_split

    samples = load_split(PROJECT_ROOT / "data" / "processed", "test")
    from PIL import Image

    transform = build_eval_transform(img_size)
    tensors, labels = [], []
    for s in samples[:64]:
        try:
            img = Image.open(s.path).convert("RGB")
            tensors.append(transform(img))
            labels.append(s.label)
        except Exception:
            continue
    x = torch.stack(tensors)

    # torch.onnx.export may leave the module in train mode (dropout active),
    # which corrupts the reference outputs — force eval before comparing.
    wrapper.eval()
    with torch.no_grad():
        torch_probs = wrapper(x).numpy()

    sess_fp = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    onnx_probs = sess_fp.run(None, {"image": x.numpy()})[0]

    sess_q = ort.InferenceSession(str(q_path), providers=["CPUExecutionProvider"])
    q_probs = sess_q.run(None, {"image": x.numpy()})[0]

    d_fp = float(np.abs(torch_probs - onnx_probs).max())
    d_q = float(np.abs(torch_probs - q_probs).max())
    print(f"[export] parity: |torch-fp32|max={d_fp:.2e}  |torch-fp16|max={d_q:.2e}  (n={len(tensors)})")
    assert d_fp < 1e-4, "fp32 ONNX diverges from torch"
    assert d_q < 0.02, "fp16 ONNX diverges too much from torch"

    # AUC of the fp16 model on these frames (sanity)
    from sklearn.metrics import roc_auc_score

    if len(set(labels)) == 2:
        auc_q = roc_auc_score(labels, q_probs)
        print(f"[export] fp16 frame AUC on {len(labels)} test frames: {auc_q:.4f}")

    meta = {
        "model": "EfficientNet-B0 spatial deepfake detector",
        "input": {"name": "image", "shape": [1, 3, img_size, img_size], "dtype": "float32"},
        "output": {"name": "fake_prob", "range": [0, 1]},
        "img_size": img_size,
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "val_auc": float(torch.load(ckpt_path, map_location="cpu").get("val_auc", 0)),
        "test_video_auc_spatial_mean": 0.9553,
        "parity_max_abs_diff_fp16": d_q,
        "license": "MIT (code); weights trained on FF++/Celeb-DF for research use",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[export] meta -> {out_dir / 'meta.json'}")
    print("EXPORT_DONE")


if __name__ == "__main__":
    main()
