#!/usr/bin/env python3
"""Cross-backend profiler for D-FINE — latency, GPU memory, and COCO mAP in one table.

Backends (pick with --backends):
  trt          our canonical engine (explicit gather-bilinear)      [C++-equivalent accuracy]
  trt-baseline the repo's grid_sample export (loses ~10 AP on TRT)  [shows our export's win]
  onnx         onnxruntime-gpu on our ONNX                          [accuracy reference == torch]
  torch        PyTorch reference (deploy+eval)                      [absolute reference]
  cpp          our C++ detector: latency via dfine_bench, mAP via dfine_coco_eval

Dataset (accuracy): --subset N (deterministic first-N sorted img_ids; presets 50/100/500/1000),
--full (all val2017), or --images DIR --ann JSON for a custom set. Latency uses a fixed dummy
input (value-independent) at each --batches size, with warm-up and p50/p90/p99.

Latency is measured per full backend call (H2D+infer+D2H) so Python backends are comparable to each
other; the `cpp` row and dfine_bench give the true, lower-overhead C++ production latency.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# This file is named profile.py, which shadows the stdlib `profile` module — and
# cProfile (pulled in lazily by torch._dynamo when the torch backend imports
# torchvision) does `import profile`, then crashes on our module. Keep this dir on
# sys.path so the sibling `coco_eval`/`cuda_env` imports resolve, but move it to the
# END so stdlib `profile` wins the name.
_scripts_dir = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _scripts_dir)]
sys.path.append(_scripts_dir)
import cv2  # noqa: E402  (imported by coco_eval too)
from coco_eval import EngineBackend, OrtBackend, TorchBackend, decode, preprocess  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SEG = Path("/home/dxdxxd/projects/custom-dfine/D-FINE-seg")
DEFAULT_LD = f"{SEG}/.venv/lib/python3.11/site-packages/tensorrt_libs:/home/dxdxxd/miniconda3/lib"


def gpu_used_mib() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / (1024 * 1024)


def percentiles(ts):
    ts = sorted(ts)
    at = lambda q: ts[min(int(q * (len(ts) - 1) + 0.5), len(ts) - 1)]
    return {"p50": at(0.50), "p90": at(0.90), "p99": at(0.99), "mean": sum(ts) / len(ts)}


# ------------------------------- latency ------------------------------------
# End-to-end per-image latency: preprocess + infer + decode, matching what the
# C++ dfine_bench measures — so the comparison is honest (Python pays the cv2
# CPU resize + numpy decode that a real deployment would, not just infer).
def latency_python(be, bgr, batch, warmup, iters, W, H, num_classes, topk, img_size):
    x_fixed = (torch.cat([preprocess(bgr, img_size) for _ in range(batch)])
               if batch > 1 else preprocess(bgr, img_size))

    def e2e():  # preprocess + infer + decode (what a real deployment pays)
        x = (torch.cat([preprocess(bgr, img_size) for _ in range(batch)])
             if batch > 1 else preprocess(bgr, img_size))
        log, box = be(x)
        for b in range(batch):
            decode(log[b:b + 1], box[b:b + 1], W, H, num_classes, topk)

    def infer():  # engine only: H2D + infer + D2H on a pre-uploaded tensor
        be(x_fixed)

    def timeit(fn):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
        return percentiles(ts)

    pe, pi = timeit(e2e), timeit(infer)
    return {"p50": pe["p50"], "p90": pe["p90"], "p99": pe["p99"], "mean": pe["mean"],
            "infer_p50": pi["p50"], "img_per_s": 1000.0 * batch / pe["p50"]}


def latency_cpp(engine, batches, warmup, iters, env, workdir, image=None):
    binj = REPO / "build" / "dfine_bench"
    out = Path(workdir) / "profile_cpp_bench.json"
    cmd = [str(binj), "--engine", engine, "--batches", ",".join(map(str, batches)),
           "--warmup", str(warmup), "--iters", str(iters), "--json", str(out)]
    if image:
        cmd += ["--image", str(image)]
    subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL)
    data = json.loads(out.read_text())
    res = {}
    for r in data["results"]:
        res[r["batch"]] = {"p50": r["total_p50"], "p90": r["total_p90"], "p99": r["total_p99"],
                           "mean": r["total_p50"], "img_per_s": r["img_per_s"],
                           "infer_p50": r["infer_p50"], "gpu_mem_mib": r["gpu_mem_mib"],
                           "stages": {"pre": r["preprocess_p50"], "infer": r["infer_p50"],
                                      "d2h": r["d2h_p50"], "decode": r["decode_p50"]}}
    return res


# ------------------------------- accuracy -----------------------------------
def accuracy_python(be, coco, img_ids, img_dir, cont2cat, num_classes, topk, score_thresh, img_size):
    results = []
    for n, iid in enumerate(img_ids):
        info = coco.loadImgs(iid)[0]
        bgr = cv2.imread(str(Path(img_dir) / info["file_name"]), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        H, W = bgr.shape[:2]
        log, box = be(preprocess(bgr, img_size))
        boxes, labels, scores = decode(log, box, W, H, num_classes, topk)
        for bb, lb, sc in zip(boxes, labels, scores):
            if sc < score_thresh:
                continue
            results.append({"image_id": iid, "category_id": cont2cat[int(lb)],
                            "bbox": [float(x) for x in bb], "score": float(sc)})
        if (n + 1) % 500 == 0:
            print(f"    ...{n + 1}/{len(img_ids)}")
    return results


def accuracy_cpp(engine, coco, img_ids, img_dir, filelist, env, score_thresh, workdir):
    binj = REPO / "build" / "dfine_coco_eval"
    out = Path(workdir) / "profile_cpp_dets.json"
    cmd = [str(binj), "--engine", engine, "--images-dir", str(img_dir), "--filelist", str(filelist),
           "--out", str(out), "--threshold", str(score_thresh)]
    subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    raw = json.loads(out.read_text())
    cats = sorted(coco.getCatIds())
    c2c = {i: c for i, c in enumerate(cats)}
    return [{"image_id": d["image_id"], "category_id": c2c[int(d["category_contig"])],
             "bbox": d["bbox"], "score": d["score"]} for d in raw]


def score_map(coco, results, img_ids):
    if not results:
        return (0.0, 0.0)
    dt = coco.loadRes(results)
    ev = COCOeval(coco, dt, iouType="bbox")
    ev.params.imgIds = img_ids
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return (ev.stats[0], ev.stats[1])


def make_backend(name, args, env):
    if name in ("trt", "trt-baseline"):
        eng = args.engine if name == "trt" else args.baseline_engine
        return EngineBackend(eng), eng
    if name == "onnx":
        return OrtBackend(args.onnx), None
    if name == "torch":
        ns = SimpleNamespace(dfine_src=args.dfine_src, model_name=args.model_name,
                             num_classes=args.num_classes, img_size=args.img_size,
                             checkpoint=args.checkpoint)
        return TorchBackend(ns), None
    raise ValueError(name)


def main(args):
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = args.ld_library_path + ":" + env.get("LD_LIBRARY_PATH", "")

    coco = COCO(args.ann)
    cats = sorted(coco.getCatIds())
    cont2cat = {i: c for i, c in enumerate(cats)}
    if args.images_from_ann:
        img_ids = sorted(coco.getImgIds())
        img_ids = img_ids if args.full else img_ids[: args.subset]
    else:
        img_ids = sorted(coco.getImgIds())[: args.subset] if not args.full else sorted(coco.getImgIds())
    print(f"[profile] backends={args.backends} batches={args.batches} "
          f"dataset={'full' if args.full else args.subset} images={len(img_ids)}")

    filelist = Path(args.workdir) / "profile_filelist.txt"
    filelist.write_text("".join(f"{i} {coco.loadImgs(i)[0]['file_name']}\n" for i in img_ids))

    # A fixed sample image drives the end-to-end latency for every backend.
    sample_path = Path(args.images) / coco.loadImgs(img_ids[0])[0]["file_name"]
    sample_bgr = cv2.imread(str(sample_path), cv2.IMREAD_COLOR)
    sH, sW = (sample_bgr.shape[0], sample_bgr.shape[1]) if sample_bgr is not None else (640, 640)

    rows = []  # (backend, batch, lat, mAP, mem)
    report = {"dataset": "full" if args.full else args.subset, "images": len(img_ids), "backends": {}}

    for name in args.backends:
        print(f"\n=== backend: {name} ===")
        entry = {"latency": {}, "map": None}
        base0 = gpu_used_mib()

        if name == "cpp":
            if args.do_latency:
                lat = latency_cpp(args.engine, args.batches, args.warmup, args.iters, env,
                                  args.workdir, sample_path)
                for B, v in lat.items():
                    entry["latency"][B] = v
                    rows.append((name, B, v, None, v.get("gpu_mem_mib")))
            if args.do_accuracy:
                res = accuracy_cpp(args.engine, coco, img_ids, args.images, filelist, env,
                                   args.score_thresh, args.workdir)
                ap, ap50 = score_map(coco, res, img_ids)
                entry["map"] = {"AP": ap, "AP50": ap50}
        else:
            be, _ = make_backend(name, args, env)
            mem_used = gpu_used_mib() - base0
            if args.do_latency:
                for B in args.batches:
                    try:
                        v = latency_python(be, sample_bgr, B, args.warmup, args.iters,
                                           sW, sH, args.num_classes, args.topk, args.img_size)
                    except Exception as e:  # e.g. batch exceeds engine max profile
                        print(f"    batch {B}: {e}")
                        continue
                    v["gpu_mem_mib"] = round(gpu_used_mib() - base0, 1)
                    entry["latency"][B] = v
                    rows.append((name, B, v, None, v["gpu_mem_mib"]))
            if args.do_accuracy:
                res = accuracy_python(be, coco, img_ids, args.images, cont2cat,
                                      args.num_classes, args.topk, args.score_thresh, args.img_size)
                ap, ap50 = score_map(coco, res, img_ids)
                entry["map"] = {"AP": ap, "AP50": ap50, "gpu_mem_mib": round(mem_used, 1)}
            del be
            torch.cuda.empty_cache()
        report["backends"][name] = entry

    # ------------------------------- report --------------------------------
    print("\n" + "=" * 78)
    if args.do_latency:
        print("latency ms: e2e = preprocess+infer+decode (real deployment);  "
              "infer = engine only (H2D+infer+D2H)")
        print(f"{'backend':<14}{'batch':>6}{'e2e p50':>9}{'infer':>8}{'e2e p90':>9}{'e2e p99':>9}"
              f"{'img/s':>9}{'GPU MiB':>9}")
        for name, B, v, _m, mem in rows:
            print(f"{name:<14}{B:>6}{v['p50']:>9.2f}{v.get('infer_p50', 0):>8.2f}{v['p90']:>9.2f}"
                  f"{v['p99']:>9.2f}{v['img_per_s']:>9.1f}{(mem if mem is not None else 0):>9.0f}")
    if args.do_accuracy:
        print("-" * 78)
        print(f"{'backend':<14}{'AP@[.5:.95]':>14}{'AP@.50':>10}")
        for name in args.backends:
            mp = report["backends"][name].get("map")
            if mp:
                print(f"{name:<14}{mp['AP']:>14.4f}{mp['AP50']:>10.4f}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\n[profile] wrote {args.out}")


def parse_args():
    p = argparse.ArgumentParser(description="D-FINE cross-backend profiler (latency + mem + mAP)")
    p.add_argument("--backends", nargs="+", default=["trt", "onnx"],
                   choices=["trt", "trt-baseline", "onnx", "torch", "cpp"])
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--subset", type=int, default=500, help="first-N sorted img_ids (deterministic)")
    p.add_argument("--full", action="store_true", help="use all val2017")
    p.add_argument("--latency", dest="do_latency", action="store_true", default=None)
    p.add_argument("--accuracy", dest="do_accuracy", action="store_true", default=None)
    p.add_argument("--no-latency", dest="do_latency", action="store_false")
    p.add_argument("--no-accuracy", dest="do_accuracy", action="store_false")
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--num-classes", type=int, default=80)
    p.add_argument("--topk", type=int, default=300)
    p.add_argument("--score-thresh", type=float, default=0.001)
    p.add_argument("--model-name", default="m")
    p.add_argument("--checkpoint", default=str(SEG / "pretrained" / "dfine_m_obj2coco.pt"))
    p.add_argument("--dfine-src", default=str(SEG))
    p.add_argument("--engine", default=str(REPO / "trt-files" / "engines" / "dfine_m_fp32.engine"))
    p.add_argument("--baseline-engine", default=str(REPO / "trt-files" / "engines" / "dfine_m_gridsample.engine"))
    p.add_argument("--onnx", default=str(REPO / "trt-files" / "onnx" / "dfine_m.onnx"))
    p.add_argument("--images", default="/mnt/d/datasets/coco/val2017")
    p.add_argument("--ann", default="/mnt/d/datasets/coco/annotations/instances_val2017.json")
    p.add_argument("--images-from-ann", action="store_true", default=True)
    p.add_argument("--out", default="")
    p.add_argument("--workdir", default=tempfile.gettempdir(), help="scratch dir for transient files")
    p.add_argument("--ld-library-path", default=DEFAULT_LD)
    a = p.parse_args()
    # Default: run both latency and accuracy unless one is explicitly toggled off.
    if a.do_latency is None:
        a.do_latency = True
    if a.do_accuracy is None:
        a.do_accuracy = True
    return a


if __name__ == "__main__":
    main(parse_args())
