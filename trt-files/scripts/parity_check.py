#!/usr/bin/env python3
"""Numerical parity for the raw D-FINE outputs: PyTorch reference vs TensorRT
(and optionally ONNXRuntime) on a real image.

The full 300-query tensor is dominated by background queries whose low-confidence
boxes are noise, so a raw all-query cosine is misleading. This compares the signal
the C++ decode actually emits: the surviving top-K detections (sigmoid + top-k over
query*class), aligned by (query, class), reporting score agreement and box L1 for
the high-confidence set — the same approach as D-FINE-seg's parity self-check.

Uses the runtime preprocessing contract (stretch to 640, /255, RGB, NCHW, no mean/std).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import torch


def preprocess(image_path: Path, img: int) -> torch.Tensor:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image_path)
    resized = cv2.resize(bgr, (img, img), interpolation=cv2.INTER_LINEAR)  # stretch
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0  # /255 only, no mean/std
    return torch.from_numpy(chw).unsqueeze(0).contiguous()


def torch_outputs(args, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    sys.path.insert(0, args.dfine_src)
    from src.d_fine.dfine import build_model
    from src.d_fine.utils import load_tuning_state

    model = build_model(args.model_name, num_classes=args.num_classes, enable_mask_head=False,
                        device="cuda", img_size=(args.img_size, args.img_size), in_channels=3,
                        pretrained_model_path=None, pretrained_backbone=False)
    model = load_tuning_state(model, args.checkpoint).cuda()
    model.deploy(); model.eval()
    with torch.no_grad():
        out = model(x.cuda())
    return out["pred_logits"].float().cpu().numpy(), out["pred_boxes"].float().cpu().numpy()


def engine_outputs(engine_path: str, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
    context = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    inp = x.cuda().contiguous()
    context.set_input_shape("images", tuple(inp.shape))
    context.set_tensor_address("images", inp.data_ptr())
    out = {}
    for name in names:
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            buf = torch.empty(tuple(context.get_tensor_shape(name)), dtype=torch.float32, device="cuda")
            out[name] = buf
            context.set_tensor_address(name, buf.data_ptr())
    stream = torch.cuda.Stream()
    context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return out["logits"].cpu().numpy(), out["boxes"].cpu().numpy()


def ort_outputs(onnx_path: str, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    o = sess.run(["logits", "boxes"], {"images": x.numpy()})
    return o[0], o[1]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def compare(name: str, ref_log, ref_box, log, box, num_classes: int, k: int) -> None:
    """Align by the reference's top-K (query,class) and compare that signal."""
    rprob = _sigmoid(ref_log[0]).reshape(-1)
    order = np.argsort(-rprob)[:k]
    q = order // num_classes
    c = order % num_classes
    ref_scores = rprob[order]
    cmp_scores = _sigmoid(log[0]).reshape(-1)[order]

    a, b = ref_scores.astype(np.float64), cmp_scores.astype(np.float64)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    max_ds = float(np.max(np.abs(a - b)))
    # box agreement for the surviving queries (cxcywh, normalized)
    box_l1 = float(np.mean(np.abs(ref_box[0, q] - box[0, q])))
    # ranking: how many of ref's top-K stay in cmp's top-K
    cmp_top = set(np.argsort(-_sigmoid(log[0]).reshape(-1))[:k].tolist())
    keep = len(set(order.tolist()) & cmp_top)
    print(f"[parity] {name:12s} top{k}: score_cos={cos:.6f}  max|dscore|={max_ds:.4f}  "
          f"box_L1={box_l1:.5f}  rank_overlap={keep}/{k}")


def main(args) -> None:
    x = preprocess(Path(args.image), args.img_size)
    print(f"[parity] image={Path(args.image).name} input={tuple(x.shape)} top-k={args.topk}")
    ref_log, ref_box = torch_outputs(args, x)

    e_log, e_box = engine_outputs(args.engine, x)
    if args.onnx and Path(args.onnx).exists():
        o_log, o_box = ort_outputs(args.onnx, x)
        compare("torch~onnx", ref_log, ref_box, o_log, o_box, args.num_classes, args.topk)
    compare("torch~trt", ref_log, ref_box, e_log, e_box, args.num_classes, args.topk)


def parse_args():
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="PyTorch vs TensorRT/ONNX parity on surviving detections")
    p.add_argument("--model-name", default="m")
    p.add_argument("--num-classes", type=int, default=80)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--topk", type=int, default=30)
    p.add_argument("--checkpoint", default="/home/dxdxxd/projects/custom-dfine/D-FINE-seg/pretrained/dfine_m_obj2coco.pt")
    p.add_argument("--dfine-src", default="/home/dxdxxd/projects/custom-dfine/D-FINE-seg")
    p.add_argument("--engine", default=str(repo / "trt-files" / "engines" / "dfine_m_fp32_notf32.engine"))
    p.add_argument("--onnx", default=str(repo / "trt-files" / "onnx" / "dfine_m.onnx"))
    p.add_argument("--image", default="/home/dxdxxd/projects/custom-dfine/D-FINE-seg/tests/assets/park_gen.jpg")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
