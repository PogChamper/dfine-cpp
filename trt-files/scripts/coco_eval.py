#!/usr/bin/env python3
"""COCO val2017 mAP for the raw D-FINE engine vs the PyTorch reference.

Turns the residual TRT-vs-PyTorch gap (M0_STATUS finding 4) into a hard number by
running both backends over COCO val and scoring with pycocotools. The decode here is
the reference the C++ M1 detector must reproduce exactly:

    sigmoid(logits) -> top-300 over (query x class) -> label=idx%C, query=idx//C
    -> cxcywh(normalized) to xyxy -> scale by original (W,H)  [stretch preprocessing]

Class ids are mapped contiguous 0..79 -> COCO category_id via the sorted category ids
(the RT-DETR/D-FINE convention).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# profile.py in this dir shadows stdlib `profile` (imported by cProfile via
# torchvision->torch._dynamo when the torch backend loads). Move the scripts dir to the
# END of sys.path so stdlib wins the name but sibling modules (cuda_env) still import.
_scripts_dir = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _scripts_dir)]
sys.path.append(_scripts_dir)

import cv2
import numpy as np
import tensorrt as trt
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def preprocess(bgr: np.ndarray, img: int) -> torch.Tensor:
    resized = cv2.resize(bgr, (img, img), interpolation=cv2.INTER_LINEAR)  # stretch
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return torch.from_numpy(chw).unsqueeze(0).contiguous()


def decode(logits: np.ndarray, boxes: np.ndarray, orig_w: int, orig_h: int,
           num_classes: int, topk: int):
    """Raw decoder outputs -> (xyxy_pixel boxes, class_idx, scores). Reference for C++ M1."""
    prob = 1.0 / (1.0 + np.exp(-logits[0]))            # [Q, C]
    flat = prob.reshape(-1)
    # topk may reach or exceed Q*C (any 1-class model at the default 300);
    # argpartition needs kth < size, so clamp — the result is simply "all".
    k = min(topk, flat.size)
    if k <= 0:
        z = np.zeros(0, dtype=np.int64)
        return np.zeros((0, 4), dtype=np.float32), z, np.zeros(0, dtype=np.float32)
    idx = np.argpartition(-flat, k - 1)[:k]
    idx = idx[np.argsort(-flat[idx])]
    scores = flat[idx]
    labels = idx % num_classes
    q = idx // num_classes
    b = boxes[0, q]                                     # [topk, 4] cxcywh normalized
    cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    x1 = (cx - 0.5 * w) * orig_w
    y1 = (cy - 0.5 * h) * orig_h
    bw = w * orig_w
    bh = h * orig_h
    return np.stack([x1, y1, bw, bh], axis=1), labels, scores


class TorchBackend:
    def __init__(self, args):
        sys.path.insert(0, args.dfine_src)
        from src.d_fine.dfine import build_model
        from src.d_fine.utils import load_tuning_state
        m = build_model(args.model_name, num_classes=args.num_classes, enable_mask_head=False,
                        device="cuda", img_size=(args.img_size, args.img_size), in_channels=3,
                        pretrained_model_path=None, pretrained_backbone=False)
        m = load_tuning_state(m, args.checkpoint).cuda()
        m.deploy(); m.eval()
        self.m = m

    @torch.no_grad()
    def __call__(self, x):
        o = self.m(x.cuda())
        return o["pred_logits"].float().cpu().numpy(), o["pred_boxes"].float().cpu().numpy()


class OrtBackend:
    def __init__(self, onnx_path):
        import cuda_env  # bootstraps onnxruntime-gpu (WSL libcuda path + preload_dlls)
        self.sess, providers = cuda_env.make_session(onnx_path)
        print(f"[coco] onnx providers: {providers}")

    def __call__(self, x):
        o = self.sess.run(["logits", "boxes"], {"images": x.numpy()})
        return o[0], o[1]


class EngineBackend:
    def __init__(self, engine_path):
        runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        self.ctx = self.engine.create_execution_context()
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.stream = torch.cuda.Stream()

    def __call__(self, x):
        inp = x.cuda().contiguous()
        self.ctx.set_input_shape("images", tuple(inp.shape))
        self.ctx.set_tensor_address("images", inp.data_ptr())
        out = {}
        for n in self.names:
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT:
                buf = torch.empty(tuple(self.ctx.get_tensor_shape(n)), dtype=torch.float32, device="cuda")
                out[n] = buf
                self.ctx.set_tensor_address(n, buf.data_ptr())
        self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return out["logits"].cpu().numpy(), out["boxes"].cpu().numpy()


def evaluate(coco: COCO, dets: list, label: str, img_ids: list) -> None:
    if not dets:
        print(f"[coco] {label}: no detections")
        return
    coco_dt = coco.loadRes(dets)
    ev = COCOeval(coco, coco_dt, iouType="bbox")
    ev.params.imgIds = img_ids  # restrict to processed images (else missing imgs count as misses)
    ev.evaluate(); ev.accumulate(); ev.summarize()
    ap, ap50 = ev.stats[0], ev.stats[1]
    print(f"[coco] {label}: AP@[.50:.95]={ap:.4f}  AP@.50={ap50:.4f}")


def main(args):
    coco = COCO(args.ann)
    cat_ids = sorted(coco.getCatIds())
    cont2cat = {i: c for i, c in enumerate(cat_ids)}      # RT-DETR/D-FINE convention
    img_ids = sorted(coco.getImgIds())
    if args.limit:
        img_ids = img_ids[: args.limit]
    print(f"[coco] images={len(img_ids)} classes={len(cat_ids)} backends={args.backends}")

    backends = {}
    if "torch" in args.backends:
        backends["torch"] = TorchBackend(args)
    if "onnx" in args.backends:
        backends["onnx"] = OrtBackend(args.onnx)
    if "engine" in args.backends:
        backends["engine"] = EngineBackend(args.engine)

    out = {k: [] for k in backends}
    img_dir = Path(args.images)
    for n, iid in enumerate(img_ids):
        info = coco.loadImgs(iid)[0]
        bgr = cv2.imread(str(img_dir / info["file_name"]), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        H, W = bgr.shape[:2]
        x = preprocess(bgr, args.img_size)
        for name, be in backends.items():
            log, box = be(x)
            boxes, labels, scores = decode(log, box, W, H, args.num_classes, args.topk)
            for bb, lb, sc in zip(boxes, labels, scores):
                if sc < args.score_thresh:
                    continue
                out[name].append({
                    "image_id": iid,
                    "category_id": cont2cat[int(lb)],
                    "bbox": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])],
                    "score": float(sc),
                })
        if (n + 1) % 200 == 0:
            print(f"[coco]   processed {n + 1}/{len(img_ids)}")

    processed = sorted({d["image_id"] for dets in out.values() for d in dets})
    for name in backends:
        evaluate(coco, out[name], name, processed)


def parse_args():
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="COCO val2017 mAP: D-FINE engine vs PyTorch")
    p.add_argument("--model-name", default="m")
    p.add_argument("--num-classes", type=int, default=80)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--topk", type=int, default=300)
    p.add_argument("--score-thresh", type=float, default=0.001)
    p.add_argument("--limit", type=int, default=0, help="0 = all val images")
    p.add_argument("--backends", nargs="+", default=["engine", "torch"], choices=["engine", "torch", "onnx"])
    p.add_argument("--checkpoint", default="/home/dxdxxd/projects/custom-dfine/D-FINE-seg/pretrained/dfine_m_obj2coco.pt")
    p.add_argument("--dfine-src", default="/home/dxdxxd/projects/custom-dfine/D-FINE-seg")
    p.add_argument("--engine", default=str(repo / "trt-files" / "engines" / "dfine_m_fp32.engine"))
    p.add_argument("--onnx", default=str(repo / "trt-files" / "onnx" / "dfine_m.onnx"))
    p.add_argument("--images", default="/mnt/d/datasets/coco/val2017")
    p.add_argument("--ann", default="/mnt/d/datasets/coco/annotations/instances_val2017.json")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
