"""
D-FINE-seg: Object Detection and Instance Segmentation Framework with Multi-backend Deployment
Copyright (c) 2026 The D-FINE-seg Authors. All Rights Reserved.

D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.

Modified by D-FINE-cpp for detection-only checkpoint export.
"""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clip(min=0.0, max=1.0)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dim=-1)


def bias_init_with_prob(prior_prob=0.01):
    return float(-math.log((1 - prior_prob) / prior_prob))


def get_activation(act: str, inplace: bool = True):
    if act is None:
        return nn.Identity()
    if isinstance(act, nn.Module):
        return act

    activations = {
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "gelu": nn.GELU,
        "hardsigmoid": nn.Hardsigmoid,
    }
    try:
        module = activations[act.lower()]()
    except KeyError as exc:
        raise ValueError(f"unsupported activation: {act}") from exc
    if hasattr(module, "inplace"):
        module.inplace = inplace
    return module


def distance2bbox(points, distance, reg_scale, deploy=False):
    if not deploy:
        reg_scale = abs(reg_scale)
    x1 = points[..., 0] - (0.5 * reg_scale + distance[..., 0]) * (points[..., 2] / reg_scale)
    y1 = points[..., 1] - (0.5 * reg_scale + distance[..., 1]) * (points[..., 3] / reg_scale)
    x2 = points[..., 0] + (0.5 * reg_scale + distance[..., 2]) * (points[..., 2] / reg_scale)
    y2 = points[..., 1] + (0.5 * reg_scale + distance[..., 3]) * (points[..., 3] / reg_scale)
    return box_xyxy_to_cxcywh(torch.stack([x1, y1, x2, y2], -1))


def weighting_function(reg_max, up, reg_scale, deploy=False):
    if deploy:
        upper_bound1 = (abs(up[0]) * abs(reg_scale)).item()
        upper_bound2 = (abs(up[0]) * abs(reg_scale) * 2).item()
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left_values = [-step**i + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values = [step**i - 1 for i in range(1, reg_max // 2)]
        values = (
            [-upper_bound2]
            + left_values
            + [torch.zeros_like(up[0][None])]
            + right_values
            + [upper_bound2]
        )
        return torch.tensor(values, dtype=up.dtype, device=up.device)

    upper_bound1 = abs(up[0]) * abs(reg_scale)
    upper_bound2 = abs(up[0]) * abs(reg_scale) * 2
    step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
    left_values = [-step**i + 1 for i in range(reg_max // 2 - 1, 0, -1)]
    right_values = [step**i - 1 for i in range(1, reg_max // 2)]
    values = (
        [-upper_bound2]
        + left_values
        + [torch.zeros_like(up[0][None])]
        + right_values
        + [upper_bound2]
    )
    return torch.cat(values, 0)


def deformable_attention_core_func_v2(
    value: torch.Tensor,
    value_spatial_shapes,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    num_points_list: List[int],
    method="default",
):
    bs, n_head, c, _ = value[0].shape
    _, length_q, _, _, _ = sampling_locations.shape
    if method == "default":
        sampling_grids = 2 * sampling_locations - 1
    elif method == "discrete":
        sampling_grids = sampling_locations
    else:
        raise ValueError(f"unsupported sampling method: {method}")

    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_locations_list = sampling_grids.split(num_points_list, dim=-2)
    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value[level].reshape(bs * n_head, c, h, w)
        sampling_grid_l = sampling_locations_list[level]
        if method == "default":
            sampling_value_l = F.grid_sample(
                value_l,
                sampling_grid_l,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        else:
            sampling_coord = (
                sampling_grid_l * torch.tensor([[w, h]], device=value_l.device) + 0.5
            ).to(torch.int64)
            sampling_coord = sampling_coord.clamp(0, h - 1)
            sampling_coord = sampling_coord.reshape(
                bs * n_head, length_q * num_points_list[level], 2
            )
            batch_idx = (
                torch.arange(sampling_coord.shape[0], device=value_l.device)
                .unsqueeze(-1)
                .repeat(1, sampling_coord.shape[1])
            )
            sampling_value_l = value_l[
                batch_idx, :, sampling_coord[..., 1], sampling_coord[..., 0]
            ]
            sampling_value_l = sampling_value_l.permute(0, 2, 1).reshape(
                bs * n_head, c, length_q, num_points_list[level]
            )
        sampling_value_list.append(sampling_value_l)

    attn_weights = attention_weights.permute(0, 2, 1, 3).reshape(
        bs * n_head, 1, length_q, sum(num_points_list)
    )
    output = (torch.concat(sampling_value_list, dim=-1) * attn_weights).sum(-1)
    return output.reshape(bs, n_head * c, length_q).permute(0, 2, 1)
