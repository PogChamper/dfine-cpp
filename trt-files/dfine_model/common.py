"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
Modifications Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.

Modified by D-FINE-cpp for checkpoint export.
"""

import torch
import torch.nn as nn


class FrozenBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.eps = eps
        self.num_features = num_features

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        state_dict.pop(prefix + "num_batches_tracked", None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        running_var = self.running_var.reshape(1, -1, 1, 1)
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        scale = weight * (running_var + self.eps).rsqrt()
        fused_bias = bias - running_mean * scale
        return x * scale + fused_bias

    def extra_repr(self):
        return f"{self.num_features}, eps={self.eps}"
