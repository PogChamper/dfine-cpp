#!/usr/bin/env python3
"""Experiment: export raw D-FINE (m) via the dynamo exporter at opset 19.

Mirrors D-FINE-seg's export path (opset 19 + dynamo=True) instead of the opset-16
legacy tracer, to test whether the different op decomposition is executed faithfully
by TensorRT (the legacy-tracer graph loses ~10 AP on TRT; see docs/impl/M0_STATUS.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn


class RawDetect(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        out = self.model(images)
        return out["pred_logits"], out["pred_boxes"]


def main(args):
    sys.path.insert(0, args.dfine_src)
    from src.d_fine.dfine import build_model
    from src.d_fine.utils import load_tuning_state

    dev = "cuda"
    m = build_model("m", num_classes=80, enable_mask_head=False, device=dev,
                    img_size=(640, 640), in_channels=3, pretrained_model_path=None, pretrained_backbone=False)
    m = load_tuning_state(m, args.checkpoint).to(dev)
    m.deploy()
    m.decoder.num_denoising = 0
    m.eval()

    dummy = torch.randn(args.trace_batch, 3, 640, 640, device=dev)
    dynamic_axes = {n: {0: "batch_size"} for n in ["images", "logits", "boxes"]}

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prog = torch.onnx.export(
        RawDetect(m), (dummy,),
        opset_version=19,
        input_names=["images"],
        output_names=["logits", "boxes"],
        dynamic_axes=dynamic_axes,
        dynamo=True,
    )
    prog.save(str(out_path))
    print(f"[dynamo] wrote {out_path}")

    import onnx
    model = onnx.load(str(out_path))
    if not args.no_simplify:
        try:
            import onnxsim
            model, ok = onnxsim.simplify(model)
            if ok:
                onnx.save(model, str(out_path))
                print("[dynamo] onnxsim simplified")
        except Exception as exc:  # noqa: BLE001
            print(f"[dynamo] onnxsim skipped: {exc}")

    model = onnx.load(str(out_path))
    gs = [n for n in model.graph.node if n.op_type == "GridSample"]
    ins = [i.name for i in model.graph.input]
    outs = [o.name for o in model.graph.output]
    print(f"[dynamo] inputs={ins} outputs={outs} GridSample={len(gs)} opset={[i.version for i in model.opset_import if i.domain in ('','ai.onnx')]}")
    for vi in list(model.graph.input) + list(model.graph.output):
        d0 = vi.type.tensor_type.shape.dim[0]
        print(f"[dynamo]   {vi.name} dim0={d0.dim_param or d0.dim_value}")


def parse_args():
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/home/dxdxxd/projects/custom-dfine/D-FINE-seg/pretrained/dfine_m_obj2coco.pt")
    p.add_argument("--dfine-src", default="/home/dxdxxd/projects/custom-dfine/D-FINE-seg")
    p.add_argument("--output", default=str(repo / "trt-files" / "onnx" / "dfine_m_op19.onnx"))
    p.add_argument("--trace-batch", type=int, default=2)
    p.add_argument("--no-simplify", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
