#!/usr/bin/env python3
"""Preprocessing A/B/C on COCO val — same engine, same run, no runtime changes.

The runtime's canonical preprocessing is stretch-resize (the convention D-FINE
was trained with). This experiment measures what the alternatives cost on the
SAME weights by preparing the input-sized canvas on the host — stretch of an
already-square canvas is an identity resample — and un-mapping the output
boxes before pycocotools scoring:

  A  stretch            canonical baseline (the runtime as shipped)
  B  letterbox-center   aspect-preserving resize (up- or downscale), centered,
                        gray-114 padding; un-map (x - dx) / s
  C  letterbox-topleft  production smart_resize semantics: NEVER upscales
                        (images smaller than the canvas are pasted 1:1),
                        top-left anchored, black padding; un-map x / s.
                        On COCO val (all images <= 640 px) this variant only
                        pads — the network sees native-scale content.

All passes share one engine and one process, scored on the same image ids.

usage:
  LD_LIBRARY_PATH=<tensorrt_libs> PYTHONPATH=python python letterbox_eval.py \
      --engine trt-files/engines/dfine_m_fp16_st_g0.engine --limit 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

REPO = Path(__file__).resolve().parents[2]


def letterbox_center(img: np.ndarray, size: int, pad: int = 114):
    h, w = img.shape[:2]
    s = min(size / w, size / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    canvas = np.full((size, size, 3), pad, np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = np.asarray(
        Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    return canvas, s, dx, dy


def letterbox_topleft(img: np.ndarray, size: int):
    # Mirrors the production smart_resize_t: no upscaling (a frame that already
    # fits is pasted 1:1), top-left anchor, black padding. (Bilinear via PIL vs
    # cv2.INTER_LINEAR — sub-pixel differences, immaterial for an mAP A/B.)
    h, w = img.shape[:2]
    canvas = np.zeros((size, size, 3), np.uint8)
    if h <= size and w <= size:
        canvas[:h, :w] = img
        return canvas, 1.0
    s = min(size / w, size / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    canvas[:nh, :nw] = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    return canvas, s


def score(coco: COCO, img_ids, results, tag: str) -> float:
    if not results:
        print(f"[{tag}] no detections")
        return 0.0
    ev = COCOeval(coco, coco.loadRes(results), iouType="bbox")
    ev.params.imgIds = img_ids  # exactly the processed subset (gotcha #8)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    print(f"[{tag}] AP@[.50:.95]={ev.stats[0]:.4f}  AP@.50={ev.stats[1]:.4f}")
    return ev.stats[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=str(REPO / "trt-files/engines/dfine_m_fp16_st_g0.engine"))
    ap.add_argument("--images", default="/mnt/d/datasets/coco/val2017")
    ap.add_argument("--ann", default="/mnt/d/datasets/coco/annotations/instances_val2017.json")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--threshold", type=float, default=0.001)
    args = ap.parse_args()

    from dfine import Detector  # needs PYTHONPATH=python + libdfine discoverable

    coco = COCO(args.ann)
    cat_ids = sorted(coco.getCatIds())
    cont2cat = {i: c for i, c in enumerate(cat_ids)}
    img_ids = sorted(coco.getImgIds())[: args.limit or None]
    print(f"[letterbox-ab] images={len(img_ids)} engine={Path(args.engine).name}")

    res_stretch: list[dict] = []
    res_center: list[dict] = []
    res_topleft: list[dict] = []

    def unmap_append(out: list[dict], iid: int, det_obj, s: float, dx: float, dy: float,
                     w: int, h: int) -> None:
        x1 = min(max((det_obj.box.x1 - dx) / s, 0.0), w)
        y1 = min(max((det_obj.box.y1 - dy) / s, 0.0), h)
        x2 = min(max((det_obj.box.x2 - dx) / s, 0.0), w)
        y2 = min(max((det_obj.box.y2 - dy) / s, 0.0), h)
        out.append({"image_id": iid, "category_id": cont2cat[det_obj.class_id],
                    "bbox": [x1, y1, x2 - x1, y2 - y1], "score": det_obj.score})

    with Detector(args.engine, threshold=args.threshold) as det:
        size = det.input_w
        assert det.input_h == size, "square input assumed"
        for n, iid in enumerate(img_ids, 1):
            info = coco.loadImgs(iid)[0]
            img = np.asarray(
                Image.open(Path(args.images) / info["file_name"]).convert("RGB"))
            h, w = img.shape[:2]

            for d in det.detect(img):  # A: canonical stretch path
                res_stretch.append({
                    "image_id": iid, "category_id": cont2cat[d.class_id],
                    "bbox": [d.box.x1, d.box.y1, d.box.x2 - d.box.x1, d.box.y2 - d.box.y1],
                    "score": d.score})

            canvas, s, dx, dy = letterbox_center(img, size)  # B
            for d in det.detect(canvas):
                unmap_append(res_center, iid, d, s, dx, dy, w, h)

            canvas, s = letterbox_topleft(img, size)  # C (prod smart_resize)
            for d in det.detect(canvas):
                unmap_append(res_topleft, iid, d, s, 0.0, 0.0, w, h)

            if n % 500 == 0:
                print(f"[letterbox-ab] {n} images...")

    ap_s = score(coco, img_ids, res_stretch, "A stretch  (canonical)")
    ap_c = score(coco, img_ids, res_center, "B letterbox center/gray")
    ap_t = score(coco, img_ids, res_topleft, "C letterbox topleft/black/no-upscale")
    print(f"\n[letterbox-ab] {len(img_ids)} images, thr={args.threshold}")
    print(f"  A stretch (canonical)                : {ap_s:.4f}")
    print(f"  B letterbox center/gray              : {ap_c:.4f}  ({ap_c - ap_s:+.4f})")
    print(f"  C letterbox topleft/black/no-upscale : {ap_t:.4f}  ({ap_t - ap_s:+.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
