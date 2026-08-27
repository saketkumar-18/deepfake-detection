"""Spatial detector: pretrained CNN backbone -> single fake logit.

The backbone is frozen-friendly: we expose two parameter groups
(head vs. backbone) so training can warm up the head first, then
fine-tune the whole network at a low LR.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


class SpatialDetector(nn.Module):
    """Frame-level binary deepfake detector.

    Args:
        backbone: timm model name (e.g. 'efficientnet_b0').
        pretrained: load ImageNet weights.
        num_classes: 1 for binary (BCE with logits).
        drop_rate: dropout before the head.
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        num_classes: int = 1,
        drop_rate: float = 0.3,
    ):
        super().__init__()
        self.net = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,          # strip classifier, keep pooling
            drop_rate=drop_rate,
        )
        self.embed_dim = self.net.num_features
        self.head = nn.Linear(self.embed_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled embedding (B, D)."""
        return self.net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits (B, num_classes)."""
        feat = self.forward_features(x)
        return self.head(feat)

    def param_groups(self, lr_backbone: float, lr_head: float, weight_decay: float):
        """Two groups: backbone (low LR) and head (high LR)."""
        return [
            {"params": list(self.net.parameters()), "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": list(self.head.parameters()), "lr": lr_head, "weight_decay": weight_decay},
        ]


def load_spatial(ckpt_path: str, backbone: str = "efficientnet_b0", map_location="cpu") -> SpatialDetector:
    """Load a SpatialDetector from a checkpoint produced by train_spatial."""
    ckpt = torch.load(ckpt_path, map_location=map_location)
    cfg = ckpt.get("config", {})
    model = SpatialDetector(
        backbone=cfg.get("backbone", backbone),
        pretrained=False,
        num_classes=cfg.get("num_classes", 1),
        drop_rate=cfg.get("drop_rate", 0.3),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model
