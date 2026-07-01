#!/usr/bin/env python3
"""Score the C++ D-FINE detector on COCO val2017 and compare to the Python reference.

Drives apps/dfine_coco_eval (C++): writes the image filelist, runs the binary to
produce COCO-style detections (contiguous class ids), maps those ids to category_id,
and scores with pycocotools. The AP should match trt-files/scripts/coco_eval.py's
engine number (m full-val ≈ 0.5507) — the only differences from that reference are
the C++ JPEG decode (stb_image) and CUDA bilinear resize.

The C++ side owns preprocessing + decode; this script only orchestrates + scores.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

REPO = Path(__file__).resolve().parents[2]
SEG = Path("/home/dxdxxd/projects/custom-dfine/D-FINE-seg")
DEFAULT_LD = (f"{SEG}/.venv/lib/python3.11/site-packages/tensorrt_libs:"
              "/home/dxdxxd/miniconda3/lib")


def main(args):
    coco = COCO(args.ann)
    cat_ids = sorted(coco.getCatIds())
    cont2cat = {i: c for i, c in enumerate(cat_ids)}  # RT-DETR/D-FINE convention
    img_ids = sorted(coco.getImgIds())
    if args.limit:
        img_ids = img_ids[: args.limit]
    print(f"[cpp_coco] images={len(img_ids)} classes={len(cat_ids)}")

    tmpdir = args.tmpdir or tempfile.gettempdir()
    filelist = Path(tmpdir) / "cpp_coco_filelist.txt"
    with open(filelist, "w") as f:
        for iid in img_ids:
            f.write(f"{iid} {coco.loadImgs(iid)[0]['file_name']}\n")

    dets_json = Path(args.out) if args.out else Path(tmpdir) / "cpp_coco_dets.json"
    cmd = [args.binary, "--engine", args.engine, "--images-dir", args.images,
           "--filelist", str(filelist), "--out", str(dets_json),
           "--threshold", str(args.score_thresh)]
    if args.meta:
        cmd += ["--meta", args.meta]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = args.ld_library_path + ":" + env.get("LD_LIBRARY_PATH", "")
    print("[cpp_coco] $", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)

    raw = json.loads(Path(dets_json).read_text())
    results = [{"image_id": d["image_id"],
                "category_id": cont2cat[int(d["category_contig"])],
                "bbox": d["bbox"], "score": d["score"]} for d in raw]
    print(f"[cpp_coco] {len(results)} detections")
    if not results:
        print("[cpp_coco] no detections — abort")
        return

    dt = coco.loadRes(results)
    ev = COCOeval(coco, dt, iouType="bbox")
    ev.params.imgIds = img_ids  # score exactly the processed subset
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    print(f"[cpp_coco] C++ detector AP@[.50:.95]={ev.stats[0]:.4f}  AP@.50={ev.stats[1]:.4f}")


def parse_args():
    p = argparse.ArgumentParser(description="COCO val2017 mAP for the C++ D-FINE detector")
    p.add_argument("--binary", default=str(REPO / "build" / "dfine_coco_eval"))
    p.add_argument("--engine", default=str(REPO / "trt-files" / "engines" / "dfine_m_fp32.engine"))
    p.add_argument("--meta", default="")
    p.add_argument("--images", default="/mnt/d/datasets/coco/val2017")
    p.add_argument("--ann", default="/mnt/d/datasets/coco/annotations/instances_val2017.json")
    p.add_argument("--limit", type=int, default=0, help="0 = all val images")
    p.add_argument("--score-thresh", type=float, default=0.001)
    p.add_argument("--out", default="")
    p.add_argument("--tmpdir", default="")
    p.add_argument("--ld-library-path", default=DEFAULT_LD)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
