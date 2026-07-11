#!/usr/bin/env python3
"""Validation spike: replace the deformable-attention grid_sample core with an
explicit gather-bilinear implementation (same math, different ops) and re-export.

If the TRT mAP recovers, the ~10 AP loss was TensorRT's fused/divergent kernel for the
grid_sample-based deformable core (which is exact in isolation but not in context), and
gather-bilinear is a plugin-free fix. If not, the divergence is in the projections and
a full MSDeformableAttention plugin is needed. See docs/impl/M0_STATUS.md.
"""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

import torch
import torch.nn as nn


def bilinear_gather(value_l, grid_l, h, w):
    # value_l [M,c,h,w]; grid_l [M,Lq,P,2] in [-1,1]; returns [M,c,Lq,P]
    M, c = value_l.shape[0], value_l.shape[1]
    Lq, P = grid_l.shape[1], grid_l.shape[2]
    gx, gy = grid_l[..., 0], grid_l[..., 1]
    ix = (gx + 1) * w / 2 - 0.5   # align_corners=False unnormalize
    iy = (gy + 1) * h / 2 - 0.5
    x0 = torch.floor(ix)
    y0 = torch.floor(iy)
    x1 = x0 + 1
    y1 = y0 + 1
    wx1 = ix - x0
    wx0 = 1 - wx1
    wy1 = iy - y0
    wy0 = 1 - wy1
    vflat = value_l.reshape(M, c, h * w)

    def corner(xc, yc, wgt):
        valid = ((xc >= 0) & (xc <= w - 1) & (yc >= 0) & (yc <= h - 1)).to(value_l.dtype)
        xcl = xc.clamp(0, w - 1)
        ycl = yc.clamp(0, h - 1)
        idx = (ycl * w + xcl).long().reshape(M, 1, Lq * P).expand(M, c, Lq * P)
        g = torch.gather(vflat, 2, idx).reshape(M, c, Lq, P)
        return g * (wgt * valid).unsqueeze(1)

    return (corner(x0, y0, wx0 * wy0) + corner(x1, y0, wx1 * wy0)
            + corner(x0, y1, wx0 * wy1) + corner(x1, y1, wx1 * wy1))


def explicit_deformable_core(value, value_spatial_shapes, sampling_locations,
                             attention_weights, num_points_list, method="default"):
    bs, n_head, c, _ = value[0].shape
    _, Len_q, _, _, _ = sampling_locations.shape
    grids = (2 * sampling_locations - 1).permute(0, 2, 1, 3, 4).flatten(0, 1)
    grids_list = grids.split(num_points_list, dim=-2)
    sampled = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value[level].reshape(bs * n_head, c, int(h), int(w))
        sampled.append(bilinear_gather(value_l, grids_list[level], int(h), int(w)))
    attn = attention_weights.permute(0, 2, 1, 3).reshape(bs * n_head, 1, Len_q, sum(num_points_list))
    out = (torch.concat(sampled, dim=-1) * attn).sum(-1).reshape(bs, n_head * c, Len_q)
    return out.permute(0, 2, 1)


class RawDetect(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        o = self.model(images)
        return o["pred_logits"], o["pred_boxes"]


def build(args):
    sys.path.insert(0, args.dfine_src)
    from src.d_fine.dfine import build_model
    from src.d_fine.utils import load_tuning_state
    m = build_model("m", num_classes=80, enable_mask_head=False, device="cuda",
                    img_size=(640, 640), in_channels=3, pretrained_model_path=None, pretrained_backbone=False)
    m = load_tuning_state(m, args.checkpoint).cuda()
    m.deploy()
    m.decoder.num_denoising = 0
    m.eval()
    return m


def patch(m):
    n = 0
    for layer in m.decoder.decoder.layers:
        layer.cross_attn.ms_deformable_attn_core = functools.partial(explicit_deformable_core, method="default")
        n += 1
    print(f"[spike] patched {n} cross_attn cores -> explicit gather-bilinear")


def main(args):
    import cv2
    m = build(args)
    img = cv2.imread("/mnt/d/datasets/coco/val2017/000000000285.jpg")
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype("float32") / 255.0).unsqueeze(0).cuda()

    with torch.no_grad():
        ref = m(x)
    patch(m)
    with torch.no_grad():
        new = m(x)
    dl = (ref["pred_logits"] - new["pred_logits"]).abs().max().item()
    db = (ref["pred_boxes"] - new["pred_boxes"]).abs().max().item()
    print(f"[spike] torch parity (orig grid_sample vs explicit): logits max_abs={dl:.3e}  boxes max_abs={db:.3e}")
    if dl > 1e-2 or db > 1e-2:
        print("[spike] WARNING: explicit core does not match grid_sample in torch — spike invalid until fixed")

    out = Path(args.output).resolve()
    dummy = torch.randn(2, 3, 640, 640, device="cuda")
    torch.onnx.export(RawDetect(m), (dummy,), str(out), input_names=["images"],
                      output_names=["logits", "boxes"],
                      dynamic_axes={n: {0: "N"} for n in ["images", "logits", "boxes"]},
                      opset_version=16, dynamo=False, training=torch.onnx.TrainingMode.EVAL)
    import onnx
    g = onnx.load(str(out)).graph
    print(f"[spike] wrote {out}: GridSample={sum(1 for n in g.node if n.op_type=='GridSample')} "
          f"Gather={sum(1 for n in g.node if n.op_type in ('Gather','GatherElements','GatherND'))}")


def parse_args():
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/home/dxdxxd/projects/custom-dfine/D-FINE-seg/pretrained/dfine_m_obj2coco.pt")
    p.add_argument("--dfine-src", default="/home/dxdxxd/projects/custom-dfine/D-FINE-seg")
    p.add_argument("--output", default=str(repo / "trt-files" / "onnx" / "dfine_m_explicit.onnx"))
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
