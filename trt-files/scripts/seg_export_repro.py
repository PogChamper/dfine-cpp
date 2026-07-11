#!/usr/bin/env python3
"""Repro for a D-FINE-seg bug report: the repo's OWN recommended export
(src/dl/export.py: opset-19 dynamo, fused postprocessor, TensorRT) loses ~10 AP on
TensorRT, and a one-function change to the deformable-attention core recovers it.

Both engines are produced by D-FINE-seg's own `export_to_onnx` + `export_to_tensorrt`
and decoded identically (the fused graph emits labels/boxes[xyxy@640]/scores). The ONLY
difference is the deformable core: stock `F.grid_sample` vs an explicit gather-bilinear
of the same math. Run on COCO val2017.

Why: TensorRT compiles the grid_sample-based deformable core divergently in-context
(GridSample is bit-exact in isolation); D-FINE's FDR box accumulation amplifies it.
D-FINE-seg's `run_parity` only checks scores (not boxes), so it doesn't catch this.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

SEG = "/home/dxdxxd/projects/custom-dfine/D-FINE-seg"
SCRIPTS = "/home/dxdxxd/projects/custom-dfine/D-FINE-cpp/trt-files/scripts"
TMP = Path(tempfile.gettempdir()) / "dfine-seg-repro"
CKPT = f"{SEG}/pretrained/dfine_m_obj2coco.pt"
IMG_DIR = Path("/mnt/d/datasets/coco/val2017")
ANN = "/mnt/d/datasets/coco/annotations/instances_val2017.json"
LIMIT = 2000
sys.path.insert(0, SEG)
sys.path.insert(0, SCRIPTS)


def build_seg_export(explicit: bool, onnx_path: Path):
    """Build the m model and export it via D-FINE-seg's OWN export functions."""
    from src.d_fine.dfine import build_model
    from src.d_fine.utils import load_tuning_state
    from src.dl.export import DFINEPostProcessor, ExportWrapper, export_to_onnx, export_to_tensorrt

    model = build_model("m", num_classes=80, enable_mask_head=False, device="cuda",
                        img_size=(640, 640), in_channels=3, pretrained_model_path=None, pretrained_backbone=False)
    model = load_tuning_state(model, CKPT).cuda()
    model.deploy()
    model.decoder.num_denoising = 0
    model.eval()
    if explicit:
        from export_dfine_onnx import patch_explicit_deform
        patch_explicit_deform(model)

    postproc = DFINEPostProcessor(num_classes=80, num_top_queries=300, use_focal_loss=True)
    wrapper = ExportWrapper(model, postproc, input_size=(640, 640)).cuda().eval()
    x_test = torch.randn(1, 3, 640, 640, device="cuda")
    # D-FINE-seg's exact export SETTINGS (opset 19, dynamo=True, fused postproc, dynamic batch),
    # but writing the ONNX directly instead of via export_to_onnx() — its onnxsim pass corrupts
    # the explicit gather graph (emits a Clip node with an unregistered input that TRT rejects).
    # onnxsim does not change the grid_sample result (stock-with-onnxsim measured the same).
    # D-FINE-seg's EXACT export (opset19, dynamo=True, onnxsim, fused postproc, dynamic batch).
    # With the explicit core's clamp expressed as min/max (export_dfine_onnx._bilinear_gather),
    # this now parses + builds for BOTH grid_sample and explicit.
    export_to_onnx(wrapper, onnx_path, x_test, max_batch_size=8, half=False,
                   dynamic_input=False, input_name="input", output_names=["labels", "boxes", "scores"])
    export_to_tensorrt(onnx_path, half=False, max_batch_size=8, opt_bs=1)
    return onnx_path.with_suffix(".engine")


def eval_fused_engine(engine_path: Path, coco: COCO, img_ids, cont2cat) -> tuple[float, float]:
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    ctx = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    in_name = next(n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
    t2t = {trt.DataType.FLOAT: torch.float32, trt.DataType.INT64: torch.int64,
           trt.DataType.INT32: torch.int32, trt.DataType.HALF: torch.float16}
    stream = torch.cuda.Stream()
    dets = []
    for iid in img_ids:
        info = coco.loadImgs(iid)[0]
        bgr = cv2.imread(str(IMG_DIR / info["file_name"]))
        if bgr is None:
            continue
        H, W = bgr.shape[:2]
        r = cv2.resize(bgr, (640, 640), interpolation=cv2.INTER_LINEAR)  # stretch, keep_ratio=False
        x = torch.from_numpy(cv2.cvtColor(r, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0).cuda().contiguous()
        ctx.set_input_shape(in_name, (1, 3, 640, 640))
        ctx.set_tensor_address(in_name, x.data_ptr())
        outs = {}
        for n in names:
            if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT:
                b = torch.empty(tuple(ctx.get_tensor_shape(n)), dtype=t2t[engine.get_tensor_dtype(n)], device="cuda")
                outs[n] = b
                ctx.set_tensor_address(n, b.data_ptr())
        ctx.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        labels = outs["labels"][0].cpu().numpy()
        boxes = outs["boxes"][0].cpu().numpy().astype(np.float64)   # xyxy in 640 space
        scores = outs["scores"][0].cpu().numpy()
        sx, sy = W / 640.0, H / 640.0
        for lb, bx, sc in zip(labels, boxes, scores):
            if sc < 0.001:
                continue
            x1, y1, x2, y2 = bx[0] * sx, bx[1] * sy, bx[2] * sx, bx[3] * sy
            dets.append({"image_id": int(iid), "category_id": cont2cat[int(lb)],
                         "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(sc)})
    if not dets:
        return 0.0, 0.0
    ev = COCOeval(coco, coco.loadRes(dets), iouType="bbox")
    ev.params.imgIds = list(img_ids)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return float(ev.stats[0]), float(ev.stats[1])


def main():
    coco = COCO(ANN)
    cont2cat = {i: c for i, c in enumerate(sorted(coco.getCatIds()))}
    img_ids = sorted(coco.getImgIds())[:LIMIT]
    print(f"[repro] D-FINE-seg export.py path (opset19/dynamo/fused), COCO {len(img_ids)} imgs, m/obj2coco")
    print("[repro] torch reference (raw decode, full-val): AP 0.5509")
    for mode in ["grid_sample", "explicit"]:
        onnx_path = TMP / f"seg_export_{mode}.onnx"
        eng = build_seg_export(explicit=(mode == "explicit"), onnx_path=onnx_path)
        ap, ap50 = eval_fused_engine(eng, coco, img_ids, cont2cat)
        tag = "STOCK D-FINE-seg" if mode == "grid_sample" else "with explicit-deform FIX"
        print(f"[repro] {mode:11s} ({tag:24s}): AP@[.50:.95]={ap:.4f}  AP@.50={ap50:.4f}")


if __name__ == "__main__":
    main()
