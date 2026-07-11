#!/usr/bin/env python3
"""Cross-backend profiler for D-FINE — latency, GPU memory, and COCO mAP in one table.

Backends (pick with --backends):
  trt          our canonical engine (explicit gather-bilinear)      [C++-equivalent accuracy]
  trt-baseline the repo's grid_sample export (loses ~10 AP on TRT)  [shows our export's win]
  onnx         onnxruntime-gpu on our ONNX                          [accuracy reference == torch]
  torch        PyTorch reference (deploy+eval)                      [absolute reference]
  cpp          our C++ detector: latency via dfine_bench, mAP via dfine_coco_eval
  cpp-graph    strict CUDA Graph replay latency (no accuracy mode)

Dataset (accuracy): --subset N (deterministic first-N sorted img_ids; presets 50/100/500/1000),
--full (all val2017), or --images DIR --ann JSON for a custom set. Latency uses the first selected
dataset image, or --sample-image when accuracy is disabled, with warm-up and p50/p90/p99.

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
from eval_contract import (  # noqa: E402
    nonnegative_int,
    positive_int,
    probability,
    require_arguments,
    require_complete_images,
    require_detection_outputs,
    require_detections,
)

REPO = Path(__file__).resolve().parents[2]


def gpu_used_mib() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / (1024 * 1024)


def percentiles(ts):
    ts = sorted(ts)

    def at(q):
        return ts[min(int(q * (len(ts) - 1) + 0.5), len(ts) - 1)]

    return {"p50": at(0.50), "p90": at(0.90), "p99": at(0.99), "mean": sum(ts) / len(ts)}


# ------------------------------- latency ------------------------------------
# End-to-end per-image latency: preprocess + infer + decode, matching what the
# C++ dfine_bench measures — so the comparison is honest (Python pays the cv2
# CPU resize + numpy decode that a real deployment would, not just infer).
def latency_python(be, bgr, batch, warmup, iters, W, H, num_classes, topk, img_size):
    x_fixed = (
        torch.cat([preprocess(bgr, img_size) for _ in range(batch)])
        if batch > 1
        else preprocess(bgr, img_size)
    )

    def e2e():  # preprocess + infer + decode (what a real deployment pays)
        x = (
            torch.cat([preprocess(bgr, img_size) for _ in range(batch)])
            if batch > 1
            else preprocess(bgr, img_size)
        )
        log, box = be(x)
        for b in range(batch):
            decode(log[b : b + 1], box[b : b + 1], W, H, num_classes, topk)

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
    return {
        "p50": pe["p50"],
        "p90": pe["p90"],
        "p99": pe["p99"],
        "mean": pe["mean"],
        "infer_p50": pi["p50"],
        "img_per_s": 1000.0 * batch / pe["p50"],
    }


def latency_cpp(engine, batches, warmup, iters, env, workdir, image=None, cuda_graph=False):
    binj = REPO / "build" / "dfine_bench"
    out = Path(workdir) / f"profile_cpp_bench{'_graph' if cuda_graph else ''}.json"
    out.unlink(missing_ok=True)
    cmd = [
        str(binj),
        "--engine",
        engine,
        "--batches",
        ",".join(map(str, batches)),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--json",
        str(out),
    ]
    if image:
        cmd += ["--image", str(image)]
    if cuda_graph:
        cmd += ["--require-cuda-graph"]
    subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL)
    data = json.loads(out.read_text())
    results = data.get("results")
    if not isinstance(results, list):
        raise RuntimeError("dfine_bench report is missing the results array")
    if not all(
        isinstance(row, dict)
        and isinstance(row.get("batch"), int)
        and not isinstance(row.get("batch"), bool)
        for row in results
    ):
        raise RuntimeError("dfine_bench report contains an invalid batch row")
    measured = [row["batch"] for row in results]
    if len(measured) != len(batches) or sorted(measured) != sorted(batches):
        raise RuntimeError(
            f"dfine_bench measured batches {measured}; expected every requested batch {batches}"
        )
    if cuda_graph and not all(row.get("cuda_graph_replay") is True for row in results):
        raise RuntimeError("dfine_bench did not confirm CUDA Graph replay for every batch")
    if cuda_graph and data.get("cuda_graph_required") is not True:
        raise RuntimeError("dfine_bench did not run in require-capture mode")
    res = {}
    for r in results:
        res[r["batch"]] = {
            "p50": r["total_p50"],
            "p90": r["total_p90"],
            "p99": r["total_p99"],
            "mean": r["total_mean"],
            "img_per_s": r["img_per_s"],
            "infer_p50": r["infer_p50"],
            "gpu_mem_mib": r["gpu_mem_mib"],
            "stages": {
                "pre": r["preprocess_p50"],
                "infer": r["infer_p50"],
                "d2h": r["d2h_p50"],
                "decode": r["decode_p50"],
            },
        }
    return res


# ------------------------------- accuracy -----------------------------------
def accuracy_python(
    be, coco, img_ids, img_dir, cont2cat, num_classes, topk, score_thresh, img_size
):
    results = []
    processed = 0
    for n, iid in enumerate(img_ids):
        info = coco.loadImgs(iid)[0]
        image_path = Path(img_dir) / info["file_name"]
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"[profile]: cannot read image: {image_path}")
        processed += 1
        H, W = bgr.shape[:2]
        log, box = be(preprocess(bgr, img_size))
        log, box = require_detection_outputs(log, box, 1, num_classes, "[profile]")
        boxes, labels, scores = decode(log, box, W, H, num_classes, topk)
        for bb, lb, sc in zip(boxes, labels, scores):
            if sc < score_thresh:
                continue
            results.append(
                {
                    "image_id": iid,
                    "category_id": cont2cat[int(lb)],
                    "bbox": [float(x) for x in bb],
                    "score": float(sc),
                }
            )
        if (n + 1) % 500 == 0:
            print(f"    ...{n + 1}/{len(img_ids)}")
    require_complete_images(len(img_ids), processed, "[profile]")
    return results


def accuracy_cpp(engine, coco, img_ids, img_dir, filelist, env, score_thresh, workdir):
    binj = REPO / "build" / "dfine_coco_eval"
    out = Path(workdir) / "profile_cpp_dets.json"
    cmd = [
        str(binj),
        "--engine",
        engine,
        "--images-dir",
        str(img_dir),
        "--filelist",
        str(filelist),
        "--out",
        str(out),
        "--threshold",
        str(score_thresh),
    ]
    subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL)
    raw = json.loads(out.read_text())
    cats = sorted(coco.getCatIds())
    c2c = {i: c for i, c in enumerate(cats)}
    return [
        {
            "image_id": d["image_id"],
            "category_id": c2c[int(d["category_contig"])],
            "bbox": d["bbox"],
            "score": d["score"],
        }
        for d in raw
    ]


def score_map(coco, results, img_ids):
    require_detections(results, "[profile]")
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
        ns = SimpleNamespace(
            dfine_src=args.dfine_src,
            model_name=args.model_name,
            num_classes=args.num_classes,
            img_size=args.img_size,
            checkpoint=args.checkpoint,
        )
        return TorchBackend(ns), None
    raise ValueError(name)


def main(args):
    env = dict(os.environ)
    library_paths = [args.ld_library_path, env.get("LD_LIBRARY_PATH", "")]
    env["LD_LIBRARY_PATH"] = ":".join(path for path in library_paths if path)
    scratch = tempfile.TemporaryDirectory(prefix="dfine-profile-", dir=args.workdir or None)
    workdir = Path(scratch.name)

    coco = None
    cont2cat = {}
    img_ids = []
    filelist = None
    sample_path = None
    sample_bgr = None
    dataset = "none"
    if args.do_accuracy:
        coco = COCO(args.ann)
        cats = sorted(coco.getCatIds())
        if len(cats) != args.num_classes:
            raise SystemExit(
                f"COCO annotations contain {len(cats)} categories; "
                f"--num-classes is {args.num_classes}"
            )
        cont2cat = {i: c for i, c in enumerate(cats)}
        img_ids = sorted(coco.getImgIds())
        img_ids = img_ids if args.full else img_ids[: args.subset]
        if not img_ids:
            raise SystemExit("COCO selection contains zero images")
        dataset = "full" if args.full else args.subset
        filelist = workdir / "filelist.txt"
        filelist.write_text("".join(f"{i} {coco.loadImgs(i)[0]['file_name']}\n" for i in img_ids))
        sample_path = Path(args.images) / coco.loadImgs(img_ids[0])[0]["file_name"]
        loaded = cv2.imread(str(sample_path), cv2.IMREAD_COLOR)
        if loaded is None:
            raise SystemExit(f"cannot read sample image: {sample_path}")
        sample_bgr = loaded
    elif args.do_latency:
        sample_path = Path(args.sample_image)
        sample_bgr = cv2.imread(str(sample_path), cv2.IMREAD_COLOR)
        if sample_bgr is None:
            raise SystemExit(f"cannot read sample image: {sample_path}")
    sH, sW = sample_bgr.shape[:2] if sample_bgr is not None else (0, 0)
    print(
        f"[profile] backends={args.backends} batches={args.batches} "
        f"dataset={dataset} images={len(img_ids)}"
    )

    rows = []  # (backend, batch, lat, mAP, mem)
    report = {"dataset": dataset, "images": len(img_ids), "backends": {}}

    for name in args.backends:
        print(f"\n=== backend: {name} ===")
        entry = {"latency": {}, "map": None}
        base0 = gpu_used_mib()

        if name in ("cpp", "cpp-graph"):
            cg = name == "cpp-graph"
            if cg and args.do_accuracy:
                raise RuntimeError("cpp-graph is latency-only; use cpp for accuracy")
            if args.do_latency:
                lat = latency_cpp(
                    args.engine,
                    args.batches,
                    args.warmup,
                    args.iters,
                    env,
                    workdir,
                    sample_path,
                    cuda_graph=cg,
                )
                for B, v in lat.items():
                    entry["latency"][B] = v
                    rows.append((name, B, v, None, v.get("gpu_mem_mib")))
            if args.do_accuracy:
                res = accuracy_cpp(
                    args.engine,
                    coco,
                    img_ids,
                    args.images,
                    filelist,
                    env,
                    args.score_thresh,
                    workdir,
                )
                ap, ap50 = score_map(coco, res, img_ids)
                entry["map"] = {"AP": ap, "AP50": ap50}
        else:
            be, _ = make_backend(name, args, env)
            mem_used = gpu_used_mib() - base0
            if args.do_latency:
                for B in args.batches:
                    v = latency_python(
                        be,
                        sample_bgr,
                        B,
                        args.warmup,
                        args.iters,
                        sW,
                        sH,
                        args.num_classes,
                        args.topk,
                        args.img_size,
                    )
                    v["gpu_mem_mib"] = round(gpu_used_mib() - base0, 1)
                    entry["latency"][B] = v
                    rows.append((name, B, v, None, v["gpu_mem_mib"]))
            if args.do_accuracy:
                res = accuracy_python(
                    be,
                    coco,
                    img_ids,
                    args.images,
                    cont2cat,
                    args.num_classes,
                    args.topk,
                    args.score_thresh,
                    args.img_size,
                )
                ap, ap50 = score_map(coco, res, img_ids)
                entry["map"] = {"AP": ap, "AP50": ap50, "gpu_mem_mib": round(mem_used, 1)}
            del be
            torch.cuda.empty_cache()
        report["backends"][name] = entry

    # ------------------------------- report --------------------------------
    print("\n" + "=" * 78)
    if args.do_latency:
        print(
            "latency ms: e2e = preprocess+infer+decode (real deployment);  "
            "infer = engine only (H2D+infer+D2H)"
        )
        print(
            f"{'backend':<14}{'batch':>6}{'e2e p50':>9}{'infer':>8}{'e2e p90':>9}{'e2e p99':>9}"
            f"{'img/s':>9}{'GPU MiB':>9}"
        )
        for name, B, v, _m, mem in rows:
            print(
                f"{name:<14}{B:>6}{v['p50']:>9.2f}{v.get('infer_p50', 0):>8.2f}{v['p90']:>9.2f}"
                f"{v['p99']:>9.2f}{v['img_per_s']:>9.1f}{(mem if mem is not None else 0):>9.0f}"
            )
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
    scratch.cleanup()
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="D-FINE cross-backend profiler (latency + mem + mAP)")
    p.add_argument(
        "--backends",
        nargs="+",
        default=["trt", "onnx"],
        choices=["trt", "trt-baseline", "onnx", "torch", "cpp", "cpp-graph"],
    )
    p.add_argument("--batches", type=positive_int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--warmup", type=nonnegative_int, default=20)
    p.add_argument("--iters", type=positive_int, default=100)
    p.add_argument(
        "--subset", type=positive_int, default=500, help="first-N sorted img_ids (deterministic)"
    )
    p.add_argument("--full", action="store_true", help="use all val2017")
    p.add_argument("--latency", dest="do_latency", action="store_true", default=None)
    p.add_argument("--accuracy", dest="do_accuracy", action="store_true", default=None)
    p.add_argument("--no-latency", dest="do_latency", action="store_false")
    p.add_argument("--no-accuracy", dest="do_accuracy", action="store_false")
    p.add_argument("--img-size", type=positive_int, default=640)
    p.add_argument("--num-classes", type=positive_int, default=80)
    p.add_argument("--topk", type=positive_int, default=300)
    p.add_argument("--score-thresh", type=probability, default=0.001)
    p.add_argument("--model-name", default="m")
    p.add_argument("--checkpoint", default=os.environ.get("DFINE_CHECKPOINT", ""))
    p.add_argument("--dfine-src", default=os.environ.get("DFINE_SEG_DIR", ""))
    p.add_argument("--engine", default=os.environ.get("ENGINE", ""))
    p.add_argument("--baseline-engine", default=os.environ.get("BASELINE_ENGINE", ""))
    p.add_argument("--onnx", default=os.environ.get("ONNX", ""))
    p.add_argument("--images", default=os.environ.get("COCO_IMAGES", ""))
    p.add_argument("--ann", default=os.environ.get("COCO_ANN", ""))
    p.add_argument(
        "--sample-image",
        default=os.environ.get("DFINE_SAMPLE_IMAGE", ""),
        help="latency input when accuracy is disabled",
    )
    p.add_argument("--out", default="")
    p.add_argument(
        "--workdir", default=tempfile.gettempdir(), help="scratch dir for transient files"
    )
    p.add_argument(
        "--ld-library-path", default="", help="prepend this directory to LD_LIBRARY_PATH"
    )
    a = p.parse_args(argv)
    # Default: run both latency and accuracy unless one is explicitly toggled off.
    if a.do_latency is None:
        a.do_latency = True
    if a.do_accuracy is None:
        a.do_accuracy = True
    if not a.do_latency and not a.do_accuracy:
        p.error("at least one of latency or accuracy must be enabled")
    if a.do_accuracy and "cpp-graph" in a.backends:
        p.error("cpp-graph is latency-only; use cpp for accuracy or pass --no-accuracy")
    if len(set(a.batches)) != len(a.batches):
        p.error("--batches must not contain duplicates")
    required = []
    if a.do_accuracy:
        required.extend([("images", "--images", "COCO_IMAGES"), ("ann", "--ann", "COCO_ANN")])
    elif a.do_latency:
        required.append(("sample_image", "--sample-image", "DFINE_SAMPLE_IMAGE"))
    if any(name in a.backends for name in ("trt", "cpp", "cpp-graph")):
        required.append(("engine", "--engine", "ENGINE"))
    if "trt-baseline" in a.backends:
        required.append(("baseline_engine", "--baseline-engine", "BASELINE_ENGINE"))
    if "onnx" in a.backends:
        required.append(("onnx", "--onnx", "ONNX"))
    if "torch" in a.backends:
        required.extend(
            [
                ("checkpoint", "--checkpoint", "DFINE_CHECKPOINT"),
                ("dfine_src", "--dfine-src", "DFINE_SEG_DIR"),
            ]
        )
    require_arguments(p, a, required)
    return a


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
