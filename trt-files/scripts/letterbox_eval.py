#!/usr/bin/env python3
"""Letterbox-vs-stretch preprocessing A/B on COCO val — same engine, same run.

The runtime's canonical preprocessing is stretch-resize (the convention D-FINE
was trained with). This experiment measures what letterbox costs on the SAME
weights: each image is letterboxed on the host into an input-sized canvas
(gray 114), fed through the standard pipeline — stretch of an already-square
canvas is an identity resample — and the output boxes are un-letterboxed
((x - dx) / s, (y - dy) / s) before pycocotools scoring. No C++ changes;
both passes share one engine and one process, scored on the same image ids.

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


def letterbox(img: np.ndarray, size: int, pad: int = 114):
    h, w = img.shape[:2]
    s = min(size / w, size / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    canvas = np.full((size, size, 3), pad, np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = np.asarray(
        Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    return canvas, s, dx, dy


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
    res_letter: list[dict] = []
    with Detector(args.engine, threshold=args.threshold) as det:
        size = det.input_w
        assert det.input_h == size, "square input assumed"
        for n, iid in enumerate(img_ids, 1):
            info = coco.loadImgs(iid)[0]
            img = np.asarray(
                Image.open(Path(args.images) / info["file_name"]).convert("RGB"))
            h, w = img.shape[:2]

            for d in det.detect(img):  # canonical stretch path
                res_stretch.append({
                    "image_id": iid, "category_id": cont2cat[d.class_id],
                    "bbox": [d.box.x1, d.box.y1, d.box.x2 - d.box.x1, d.box.y2 - d.box.y1],
                    "score": d.score})

            canvas, s, dx, dy = letterbox(img, size)
            for d in det.detect(canvas):  # identity resample of the letterboxed canvas
                x1 = min(max((d.box.x1 - dx) / s, 0.0), w)
                y1 = min(max((d.box.y1 - dy) / s, 0.0), h)
                x2 = min(max((d.box.x2 - dx) / s, 0.0), w)
                y2 = min(max((d.box.y2 - dy) / s, 0.0), h)
                res_letter.append({
                    "image_id": iid, "category_id": cont2cat[d.class_id],
                    "bbox": [x1, y1, x2 - x1, y2 - y1], "score": d.score})
            if n % 500 == 0:
                print(f"[letterbox-ab] {n} images...")

    ap_s = score(coco, img_ids, res_stretch, "stretch  (canonical)")
    ap_l = score(coco, img_ids, res_letter, "letterbox (host A/B) ")
    print(f"\n[letterbox-ab] delta = {ap_l - ap_s:+.4f} AP "
          f"({(ap_l - ap_s) / ap_s * 100 if ap_s else 0:+.2f}% vs stretch) — "
          f"{len(img_ids)} images, thr={args.threshold}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
