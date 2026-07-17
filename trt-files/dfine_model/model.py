"""
D-FINE detection model assembly for checkpoint export.

Derived from D-FINE-seg at the revision recorded in trt-files/DFINE_SEG_REVISION.
Copyright (c) 2026 The D-FINE-seg Authors. All Rights Reserved.
Modified by D-FINE-cpp for detection-only checkpoint export.
"""

from copy import deepcopy

import torch.nn as nn

from .configs import models
from .dfine_decoder import DFINETransformer
from .hgnetv2 import HGNetv2
from .hybrid_encoder import HybridEncoder


class DFINE(nn.Module):
    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, images):
        return self.decoder(self.encoder(self.backbone(images)))

    def deploy(self):
        self.eval()
        for module in self.modules():
            if hasattr(module, "convert_to_deploy"):
                module.convert_to_deploy()
        return self


def build_model(model_name: str, num_classes: int, img_size: tuple[int, int]) -> DFINE:
    try:
        config = deepcopy(models[model_name])
    except KeyError as exc:
        raise ValueError(f"unsupported D-FINE variant: {model_name}") from exc
    config["HybridEncoder"]["eval_spatial_size"] = img_size
    config["DFINETransformer"]["eval_spatial_size"] = img_size
    return DFINE(
        HGNetv2(in_channels=3, **config["HGNetv2"]),
        HybridEncoder(**config["HybridEncoder"]),
        DFINETransformer(num_classes=num_classes, **config["DFINETransformer"]),
    )
