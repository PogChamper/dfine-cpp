#!/usr/bin/env python3
"""Cross-backend profiler for D-FINE — latency, GPU memory, and COCO bbox metrics.

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

Latency scopes are reported separately. TensorRT exposes device-resident engine time,
transfer-inclusive forward time, and full image-to-detections time. PyTorch and ONNX Runtime expose
host-to-host backend-call time and full image-to-detections time. Native C++ stages retain the exact
fields reported by dfine_bench.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import median
from types import SimpleNamespace

import torch
from pycocotools.coco import COCO

# This file is named profile.py, which shadows the stdlib `profile` module — and
# cProfile (pulled in lazily by torch._dynamo when the torch backend imports
# torchvision) does `import profile`, then crashes on our module. Keep this dir on
# sys.path so the sibling `coco_eval`/`cuda_env` imports resolve, but move it to the
# END so stdlib `profile` wins the name.
_scripts_dir = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _scripts_dir)]
sys.path.append(_scripts_dir)
import cv2  # noqa: E402  (imported by coco_eval too)
import numpy as np  # noqa: E402
from coco_eval import (  # noqa: E402
    EngineBackend,
    OrtBackend,
    TorchBackend,
    _artifact_contract,
    _artifact_lineage,
    _require_engine_source,
    _require_evaluation_arguments,
    _require_matching_model_contracts,
    decode,
    preprocess,
)
from coco_metrics import evaluate_bbox, ground_truth_summary  # noqa: E402
from eval_contract import (  # noqa: E402
    nonnegative_int,
    positive_int,
    probability,
    require_arguments,
    require_complete_images,
    require_detection_outputs,
    require_detections,
)
from evaluation_report import (  # noqa: E402
    artifact,
    atomic_json,
    discovered_sidecar,
    environment_metadata,
    evaluation_contract,
    package_runtime,
    paths_alias,
    sha256_file,
    sidecar_recipe,
)

REPO = Path(__file__).resolve().parents[2]
ROUND_AGGREGATION = "median_across_independent_rounds"
NATIVE_BENCH = REPO / "build" / "dfine_bench"
NATIVE_COCO_EVAL = REPO / "build" / "dfine_coco_eval"

LATENCY_SCOPE_DEFINITIONS = {
    "end_to_end": (
        "CPU BGR image to decoded CPU detections: preprocessing, H2D, forward, D2H, and decode"
    ),
    "backend_call_host_to_host": (
        "preprocessed CPU tensor to CPU logits and boxes through the backend API; includes "
        "framework overhead and transfers"
    ),
    "trt_transfer_inclusive": (
        "pinned host input to pinned host outputs: H2D, TensorRT enqueueV3, and D2H; "
        "CUDA events with preallocated buffers"
    ),
    "trt_engine_device": (
        "resident device input to resident device outputs: TensorRT enqueueV3; "
        "CUDA events with prebound buffers"
    ),
    "cuda_graph_engine_d2h": ("native CUDA Graph replay containing TensorRT enqueueV3 and D2H"),
}

ENGINE_CONTRACT_FIELDS = (
    "model",
    "variant",
    "input_h",
    "input_w",
    "num_classes",
    "num_queries",
    "eval_idx",
    "cascade",
    "cascade_initial_queries",
    "resize",
    "precision",
    "precision_mode",
    "network_typing",
    "tf32",
    "dynamic_batch",
    "min_batch",
    "opt_batch",
    "max_batch",
    "max_aux_streams",
    "cuda_graph_compat",
    "trt_version",
    "sm_arch",
    "onnx_sha256",
)
REQUIRED_ENGINE_CONTRACT_FIELDS = (
    "precision",
    "precision_mode",
    "network_typing",
    "tf32",
    "dynamic_batch",
    "min_batch",
    "opt_batch",
    "max_batch",
    "max_aux_streams",
    "cuda_graph_compat",
    "trt_version",
    "sm_arch",
    "onnx_sha256",
)


def gpu_used_mib() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / (1024 * 1024)


def active_gpu_identity(environment: dict) -> dict:
    logical_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(logical_index)
    raw_uuid = str(getattr(properties, "uuid", "")) or None
    uuid = (
        raw_uuid if raw_uuid is None or raw_uuid.startswith(("GPU-", "MIG-")) else f"GPU-{raw_uuid}"
    )
    identity = {
        "logical_cuda_index": logical_index,
        "name": properties.name,
        "uuid": uuid,
        "memory_bytes": properties.total_memory,
        "compute_capability": f"{properties.major}.{properties.minor}",
    }
    physical = next(
        (gpu for gpu in environment.get("nvidia_gpus") or [] if gpu["uuid"] == uuid), None
    )
    if physical is not None:
        identity["nvidia_smi"] = physical
    return identity


def native_gpu_identity(environment: dict) -> dict | None:
    """Resolve native CUDA device 0 without creating a parent CUDA context."""
    gpus = environment.get("nvidia_gpus") or []
    if not gpus:
        return None
    visible = environment.get("cuda_visible_devices")
    if visible is not None and not visible.strip():
        return None
    token = visible.split(",", 1)[0].strip() if visible else "0"
    physical = next(
        (
            gpu
            for gpu in gpus
            if str(gpu["index"]) == token
            or gpu["uuid"] == token
            or gpu["uuid"].startswith(token)
        ),
        None,
    )
    if physical is None:
        return None
    return {
        "logical_cuda_index": 0,
        "name": physical["name"],
        "uuid": physical["uuid"],
        "memory_bytes": physical["memory_mib"] * 1024 * 1024,
        "nvidia_smi": physical,
    }


def percentiles(ts):
    ts = sorted(ts)
    if not ts or any(not math.isfinite(value) or value < 0.0 for value in ts):
        raise RuntimeError("latency samples must be finite and nonnegative")

    def at(q):
        return ts[min(int(q * (len(ts) - 1) + 0.5), len(ts) - 1)]

    return {"p50": at(0.50), "p90": at(0.90), "p99": at(0.99), "mean": sum(ts) / len(ts)}


def _metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


# ------------------------------- latency ------------------------------------
def _with_throughput(samples, batch):
    result = percentiles(samples)
    if result["p50"] <= 0.0:
        raise RuntimeError("median latency must be greater than zero")
    result["img_per_s"] = 1000.0 * batch / result["p50"]
    return result


def _aggregate_rounds(values: list[dict]) -> dict:
    if not values:
        raise ValueError("at least one measurement round is required")
    keys = set(values[0])
    if any(set(value) != keys for value in values):
        raise RuntimeError("measurement rounds expose inconsistent fields")
    result = {}
    for key in sorted(keys):
        samples = [value[key] for value in values]
        if any(
            not isinstance(sample, (int, float))
            or isinstance(sample, bool)
            or not math.isfinite(sample)
            or sample < 0
            for sample in samples
        ):
            raise RuntimeError(f"measurement rounds contain invalid {key}")
        result[key] = median(samples)
    result["aggregation"] = ROUND_AGGREGATION
    result["rounds"] = values
    return result


def _measure_rounds(measure, rounds: int) -> dict:
    return _aggregate_rounds([measure() for _ in range(rounds)])


def _wall_latency(fn, warmup, iters, batch):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    return _with_throughput(samples, batch)


def _cuda_latency(stream, fn, warmup, iters, batch):
    for _ in range(warmup):
        fn()
    stream.synchronize()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iters)
    ]
    for start, end in events:
        start.record(stream)
        end.record(stream)
    stream.synchronize()
    for start, end in events:
        start.record(stream)
        fn()
        end.record(stream)
    stream.synchronize()
    return _with_throughput([start.elapsed_time(end) for start, end in events], batch)


def _validate_outputs(logits, boxes, batch, num_classes, label):
    require_detection_outputs(logits, boxes, batch, num_classes, label)


def latency_python(
    be,
    bgr,
    batch,
    warmup,
    iters,
    W,
    H,
    num_classes,
    topk,
    img_size,
    score_thresh=0.0,
    rounds=1,
):
    x_fixed = (
        torch.cat([preprocess(bgr, img_size) for _ in range(batch)])
        if batch > 1
        else preprocess(bgr, img_size)
    )

    def e2e():  # CPU BGR input to decoded CPU detections
        x = (
            torch.cat([preprocess(bgr, img_size) for _ in range(batch)])
            if batch > 1
            else preprocess(bgr, img_size)
        )
        log, box = be(x)
        for b in range(batch):
            boxes, labels, scores = decode(log[b : b + 1], box[b : b + 1], W, H, num_classes, topk)
            keep = scores >= score_thresh
            _ = (
                np.ascontiguousarray(boxes[keep]),
                np.ascontiguousarray(labels[keep]),
                np.ascontiguousarray(scores[keep]),
            )

    initial_logits, initial_boxes = be(x_fixed)
    _validate_outputs(initial_logits, initial_boxes, batch, num_classes, "[profile] backend")
    scopes = {
        "end_to_end": _measure_rounds(lambda: _wall_latency(e2e, warmup, iters, batch), rounds)
    }

    if isinstance(be, EngineBackend):
        call = be.reusable_call(x_fixed)
        call.upload()
        call.synchronize()
        scopes["trt_engine_device"] = _measure_rounds(
            lambda: _cuda_latency(call.stream, call.enqueue, warmup, iters, batch), rounds
        )
        call.require_finite(device=True, host=False)
        scopes["trt_transfer_inclusive"] = _measure_rounds(
            lambda: _cuda_latency(call.stream, call.transfer, warmup, iters, batch), rounds
        )
        call.require_finite(device=True, host=True)
        logits, boxes = call.numpy_outputs()
        _validate_outputs(logits, boxes, batch, num_classes, "[profile] TensorRT")
    else:
        scopes["backend_call_host_to_host"] = _measure_rounds(
            lambda: _wall_latency(lambda: be(x_fixed), warmup, iters, batch), rounds
        )
    final_logits, final_boxes = be(x_fixed)
    _validate_outputs(final_logits, final_boxes, batch, num_classes, "[profile] backend")
    return {"scopes": scopes}


def _latency_cpp_round(
    engine,
    batches,
    warmup,
    iters,
    env,
    workdir,
    image=None,
    cuda_graph=False,
    round_index=0,
    score_thresh=0.001,
    input_wh=None,
):
    binj = NATIVE_BENCH
    out = (
        Path(workdir)
        / f"profile_cpp_bench{'_graph' if cuda_graph else ''}_round{round_index + 1}.json"
    )
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
        "--threshold",
        str(score_thresh),
    ]
    if image:
        cmd += ["--image", str(image)]
    if cuda_graph:
        cmd += ["--require-cuda-graph"]
    subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL)
    data = json.loads(out.read_text())
    if not isinstance(data, dict):
        raise RuntimeError("dfine_bench report root must be an object")
    reported_engine = data.get("engine")
    if not isinstance(reported_engine, str) or not paths_alias(reported_engine, engine):
        raise RuntimeError("dfine_bench report engine does not match the requested engine")
    reported_input = data.get("input")
    if (
        not isinstance(reported_input, list)
        or len(reported_input) != 2
        or any(type(value) is not int or value <= 0 for value in reported_input)
    ):
        raise RuntimeError("dfine_bench report contains invalid input geometry")
    if input_wh is not None and tuple(reported_input) != tuple(input_wh):
        raise RuntimeError(
            f"dfine_bench measured input {reported_input}; expected {list(input_wh)}"
        )
    if data.get("cuda_graph") is not cuda_graph:
        raise RuntimeError("dfine_bench report CUDA Graph mode does not match the request")
    if data.get("cuda_graph_required") is not cuda_graph:
        raise RuntimeError("dfine_bench report require-capture mode does not match the request")
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
    latency_fields = (
        "total_p50",
        "total_mean",
        "total_p90",
        "total_p99",
        "preprocess_p50",
        "infer_p50",
        "d2h_p50",
        "decode_p50",
        "img_per_s",
        "gpu_mem_mib",
    )
    for row in results:
        if not isinstance(row.get("cuda_graph_replay"), bool):
            raise RuntimeError("dfine_bench report is missing the CUDA Graph state")
        for field in latency_fields:
            value = row.get(field)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise RuntimeError(f"dfine_bench reported invalid {field}")
        p50 = float(row["total_p50"])
        p90 = float(row["total_p90"])
        p99 = float(row["total_p99"])
        rate = float(row["img_per_s"])
        if p50 <= 0.0 or rate <= 0.0 or not p50 <= p90 <= p99:
            raise RuntimeError("dfine_bench reported inconsistent latency percentiles")
        expected_rate = 1000.0 * row["batch"] / p50
        if not math.isclose(rate, expected_rate, rel_tol=0.01, abs_tol=0.05):
            raise RuntimeError("dfine_bench throughput is inconsistent with median latency")
    res = {}
    for r in results:
        total = {
            "p50": r["total_p50"],
            "p90": r["total_p90"],
            "p99": r["total_p99"],
            "mean": r["total_mean"],
            "img_per_s": r["img_per_s"],
        }
        stages = {
            "preprocess_device": {"p50": r["preprocess_p50"]},
            "decode_host": {"p50": r["decode_p50"]},
        }
        scopes = {"end_to_end": total}
        if r.get("cuda_graph_replay") is True:
            stages["cuda_graph_engine_d2h"] = {"p50": r["infer_p50"]}
            scopes["cuda_graph_engine_d2h"] = {"p50": r["infer_p50"]}
        else:
            stages["trt_engine_device"] = {"p50": r["infer_p50"]}
            stages["device_to_host"] = {"p50": r["d2h_p50"]}
            scopes["trt_engine_device"] = {"p50": r["infer_p50"]}
        res[r["batch"]] = {
            "scopes": scopes,
            "gpu_mem_mib": r["gpu_mem_mib"],
            "native_stages": stages,
        }
    return res


def _aggregate_native_rounds(values: list[dict], batches: list[int]) -> dict:
    result = {}
    for batch in batches:
        rows = [value[batch] for value in values]
        scope_names = set(rows[0]["scopes"])
        stage_names = set(rows[0]["native_stages"])
        if any(set(row["scopes"]) != scope_names for row in rows):
            raise RuntimeError("native rounds expose inconsistent latency scopes")
        if any(set(row["native_stages"]) != stage_names for row in rows):
            raise RuntimeError("native rounds expose inconsistent stage metrics")
        result[batch] = {
            "scopes": {
                name: _aggregate_rounds([row["scopes"][name] for row in rows])
                for name in sorted(scope_names)
            },
            "gpu_mem_mib": median(row["gpu_mem_mib"] for row in rows),
            "gpu_mem_mib_rounds": [row["gpu_mem_mib"] for row in rows],
            "native_stages": {
                name: _aggregate_rounds([row["native_stages"][name] for row in rows])
                for name in sorted(stage_names)
            },
        }
    return result


def latency_cpp(
    engine,
    batches,
    warmup,
    iters,
    env,
    workdir,
    image=None,
    cuda_graph=False,
    rounds=1,
    score_thresh=0.001,
    input_wh=None,
):
    values = [
        _latency_cpp_round(
            engine,
            batches,
            warmup,
            iters,
            env,
            workdir,
            image=image,
            cuda_graph=cuda_graph,
            round_index=round_index,
            score_thresh=score_thresh,
            input_wh=input_wh,
        )
        for round_index in range(rounds)
    ]
    return _aggregate_native_rounds(values, batches)


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


def accuracy_cpp(engine, coco, img_ids, img_dir, filelist, env, score_thresh, topk, workdir):
    binj = NATIVE_COCO_EVAL
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
    dets = [
        {
            "image_id": d["image_id"],
            "category_id": c2c[int(d["category_contig"])],
            "bbox": d["bbox"],
            "score": d["score"],
        }
        for d in raw
    ]
    # The native runtime decodes top-min(300, candidates) per image; the shared
    # evaluation contract records args.topk for every backend, so enforce it here
    # by per-image score truncation (identical selection order to decode()).
    per_image: dict = {}
    for d in dets:
        per_image.setdefault(d["image_id"], []).append(d)
    truncated = []
    for img_dets in per_image.values():
        img_dets.sort(key=lambda d: d["score"], reverse=True)
        truncated.extend(img_dets[:topk])
    return truncated


def _resolved_libdfine(executable: Path, env: dict) -> Path:
    query = subprocess.run(
        ["ldd", str(executable)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if query.returncode != 0:
        raise RuntimeError(f"cannot inspect native dependencies for {executable}: {query.stderr}")
    for line in query.stdout.splitlines():
        left, separator, right = line.partition("=>")
        if not left.strip().startswith("libdfine.so"):
            continue
        if not separator or right.strip().startswith("not found"):
            raise RuntimeError(f"cannot resolve libdfine for {executable}")
        candidate = Path(right.strip().split(" (", 1)[0]).resolve()
        if not candidate.is_file():
            raise RuntimeError(f"resolved libdfine does not exist: {candidate}")
        return candidate
    raise RuntimeError(f"{executable} does not load a shared libdfine runtime")


def _native_runtime_artifacts(executables: list[Path], env: dict) -> dict:
    executable_records = [
        artifact(
            "native_executable",
            executable,
            recipe=executable.name,
            runtime="D-FINE-cpp native benchmark",
        )
        for executable in executables
    ]
    libraries = {_resolved_libdfine(executable, env) for executable in executables}
    if len(libraries) != 1:
        raise RuntimeError("native benchmark executables resolve different libdfine libraries")
    library = libraries.pop()
    return {
        "executables": executable_records,
        "libdfine": artifact(
            "native_shared_library",
            library,
            recipe="loaded by benchmark executable",
            runtime="D-FINE-cpp native runtime",
        ),
    }


def score_map(coco, results, img_ids, img_size=640):
    require_detections(results, "[profile]")
    return evaluate_bbox(coco, results, img_ids, model_hw=(img_size, img_size))


def make_backend(name, args, env):
    if name in ("trt", "trt-baseline"):
        eng = args.engine if name == "trt" else args.baseline_engine
        return EngineBackend(eng), eng
    if name == "onnx":
        return OrtBackend(args.onnx, allow_cpu=getattr(args, "allow_onnx_cpu", False)), None
    if name == "torch":
        ns = SimpleNamespace(
            model_name=args.model_name,
            num_classes=args.num_classes,
            img_size=args.img_size,
            checkpoint=args.checkpoint,
            num_queries=None,
            eval_idx=None,
            cascade=None,
        )
        return TorchBackend(ns), None
    raise ValueError(name)


def backend_artifact(name, args) -> dict:
    if name == "torch":
        return artifact(
            "checkpoint",
            args.checkpoint,
            recipe=f"eager-fp32;model={args.model_name};queries=300",
            runtime=package_runtime("PyTorch", "torch"),
        )
    if name == "onnx":
        path = args.onnx
        return artifact(
            "onnx",
            path,
            recipe=sidecar_recipe(path, "explicit-fp32"),
            runtime=package_runtime("ONNX Runtime", "onnxruntime-gpu"),
            sidecar=discovered_sidecar(path),
        )
    path = args.baseline_engine if name == "trt-baseline" else args.engine
    runtime = "D-FINE-cpp + " if name in {"cpp", "cpp-graph"} else ""
    return artifact(
        "tensorrt_engine",
        path,
        recipe=sidecar_recipe(path, "TensorRT engine"),
        runtime=package_runtime(f"{runtime}TensorRT", "tensorrt"),
        sidecar=discovered_sidecar(path),
    )


def _validated_engine_contract(metadata: dict, sidecar: Path, args) -> dict:
    missing = [field for field in REQUIRED_ENGINE_CONTRACT_FIELDS if field not in metadata]
    if missing:
        raise RuntimeError(f"engine sidecar {sidecar} is missing: {', '.join(missing)}")
    for field in ("precision", "precision_mode", "network_typing", "trt_version", "sm_arch"):
        if not isinstance(metadata[field], str) or not metadata[field]:
            raise RuntimeError(f"engine sidecar {sidecar}: {field} must be a non-empty string")
    for field in ("tf32", "dynamic_batch", "cuda_graph_compat"):
        if type(metadata[field]) is not bool:
            raise RuntimeError(f"engine sidecar {sidecar}: {field} must be a boolean")
    batches = []
    for field in ("min_batch", "opt_batch", "max_batch"):
        value = metadata[field]
        if type(value) is not int or value <= 0:
            raise RuntimeError(f"engine sidecar {sidecar}: {field} must be a positive integer")
        batches.append(value)
    if batches != sorted(batches):
        raise RuntimeError(f"engine sidecar {sidecar}: batch profile must satisfy min <= opt <= max")
    if any(batch < batches[0] or batch > batches[2] for batch in args.batches):
        raise RuntimeError(
            f"engine sidecar {sidecar}: requested batches {args.batches} exceed "
            f"profile {batches[0]}/{batches[1]}/{batches[2]}"
        )
    streams = metadata["max_aux_streams"]
    if streams is not None and (type(streams) is not int or streams < 0):
        raise RuntimeError(
            f"engine sidecar {sidecar}: max_aux_streams must be null or a non-negative integer"
        )
    return {
        "sidecar": str(sidecar.resolve()),
        "sidecar_sha256": sha256_file(sidecar),
        **{field: metadata[field] for field in ENGINE_CONTRACT_FIELDS if field in metadata},
    }


def validate_backend_contracts(args) -> tuple[dict[str, dict], dict[str, dict], dict[str, dict]]:
    paths = {}
    contract_backends = {}
    if any(name in args.backends for name in ("trt", "cpp", "cpp-graph")):
        paths["engine"] = args.engine
        contract_backends["engine"] = [
            name for name in args.backends if name in {"trt", "cpp", "cpp-graph"}
        ]
    if "trt-baseline" in args.backends:
        paths["baseline engine"] = args.baseline_engine
        contract_backends["baseline engine"] = ["trt-baseline"]
    if "onnx" in args.backends:
        paths["ONNX graph"] = args.onnx

    engine_contracts = {}
    model_contracts = {}
    lineages = {}
    for label, source in paths.items():
        try:
            kind = "onnx" if label == "ONNX graph" else "engine"
            model_contract, metadata, sidecar = _artifact_contract(source, kind)
            _require_evaluation_arguments(model_contract, args, label)
            lineage = _artifact_lineage(source, kind, metadata, sidecar, model_contract)
        except SystemExit as exc:
            raise RuntimeError(str(exc)) from exc

        backends = ["onnx"] if label == "ONNX graph" else contract_backends[label]
        for backend in backends:
            model_contracts[backend] = model_contract
            lineages[backend] = lineage
        if label in contract_backends:
            contract = _validated_engine_contract(metadata, sidecar, args)
            for backend in contract_backends[label]:
                engine_contracts[backend] = contract

    if model_contracts:
        try:
            _require_matching_model_contracts(model_contracts)
            for backend in ("trt", "cpp", "cpp-graph"):
                if "onnx" not in lineages or backend not in lineages:
                    continue
                if lineages["onnx"]["precision_mode"] != lineages[backend]["precision_mode"]:
                    raise RuntimeError(f"ONNX and {backend} precision modes differ")
                _require_engine_source(lineages[backend], args.onnx)
        except SystemExit as exc:
            raise RuntimeError(str(exc)) from exc
    return engine_contracts, model_contracts, lineages


def main(args):
    engine_contracts, model_contracts, lineages = validate_backend_contracts(args)
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
    report = {
        "schema": 2,
        "dataset": dataset,
        "images": len(img_ids),
        "latency_scope_definitions": LATENCY_SCOPE_DEFINITIONS,
        "backends": {},
    }
    environment = environment_metadata()
    if args.do_latency:
        python_cuda = any(name not in {"cpp", "cpp-graph"} for name in args.backends)
        report["latency_protocol"] = {
            "batches": args.batches,
            "warmup": args.warmup,
            "iters": args.iters,
            "rounds": args.rounds,
            "aggregation": ROUND_AGGREGATION,
            "score_threshold": args.score_thresh,
            "topk": args.topk,
            "sample": {
                "path": str(sample_path.resolve()),
                "sha256": sha256_file(sample_path),
                "width": int(sample_bgr.shape[1]),
                "height": int(sample_bgr.shape[0]),
            },
            "gpu_identity": (
                active_gpu_identity(environment)
                if python_cuda
                else native_gpu_identity(environment)
            ),
        }
    if args.do_accuracy:
        report["ground_truth"] = ground_truth_summary(coco, img_ids)
        report["evaluation_contract"] = evaluation_contract(
            coco,
            img_ids,
            args.images,
            args.ann,
            score_threshold=args.score_thresh,
            topk=args.topk,
            inference_batch_size=1,
            model_hw=(args.img_size, args.img_size),
            metrics_source=Path(__file__).with_name("coco_metrics.py"),
        )
    report["provenance"] = {
        "script_sha256": sha256_file(Path(__file__)),
        "metrics_sha256": sha256_file(Path(__file__).with_name("coco_metrics.py")),
        "protocol_manifest": (
            {
                "path": str(Path(args.protocol_manifest).resolve()),
                "sha256": sha256_file(args.protocol_manifest),
            }
            if args.protocol_manifest
            else None
        ),
        "environment": environment,
    }

    for name in args.backends:
        print(f"\n=== backend: {name} ===")
        entry = {"artifact": backend_artifact(name, args), "latency": {}, "map": None}
        if name in engine_contracts:
            entry["engine_contract"] = engine_contracts[name]
        if name in model_contracts:
            entry["model_contract"] = model_contracts[name]
            entry["lineage"] = lineages[name]

        if name in ("cpp", "cpp-graph"):
            cg = name == "cpp-graph"
            native_executables = []
            if args.do_latency:
                native_executables.append(NATIVE_BENCH)
            if args.do_accuracy:
                native_executables.append(NATIVE_COCO_EVAL)
            entry["runtime_artifacts"] = _native_runtime_artifacts(native_executables, env)
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
                    rounds=args.rounds,
                    score_thresh=args.score_thresh,
                    input_wh=(args.img_size, args.img_size),
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
                    args.topk,
                    workdir,
                )
                entry["map"] = score_map(coco, res, img_ids, args.img_size)
            if entry["runtime_artifacts"] != _native_runtime_artifacts(native_executables, env):
                raise RuntimeError("native runtime artifacts changed during the benchmark")
        else:
            base0 = gpu_used_mib()
            be, _ = make_backend(name, args, env)
            backend_model_contract = getattr(be, "model_contract", None)
            backend_lineage = getattr(be, "lineage", None)
            if (backend_model_contract is None) != (backend_lineage is None):
                raise RuntimeError(f"{name} backend returned incomplete model lineage")
            if backend_model_contract is not None:
                compared = {**model_contracts, name: backend_model_contract}
                try:
                    _require_matching_model_contracts(compared)
                except SystemExit as exc:
                    raise RuntimeError(str(exc)) from exc
                entry["model_contract"] = backend_model_contract
                entry["lineage"] = backend_lineage
            backend_provenance = getattr(be, "provenance", None)
            if backend_provenance is not None:
                entry["backend_provenance"] = backend_provenance
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
                        args.score_thresh,
                        args.rounds,
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
                entry["map"] = score_map(coco, res, img_ids, args.img_size)
                entry["map"]["gpu_mem_mib"] = round(mem_used, 1)
            del be
            torch.cuda.empty_cache()
        report["backends"][name] = entry

    # ------------------------------- report --------------------------------
    print("\n" + "=" * 78)
    if args.do_latency:
        print("latency ms by measurement scope:")
        print(
            f"{'backend':<14}{'batch':>6}  {'scope':<27}{'p50':>9}{'p90':>9}{'p99':>9}{'img/s':>10}{'GPU MiB':>9}"
        )
        for name, B, v, _m, mem in rows:
            for scope, metrics in v["scopes"].items():
                p90 = metrics.get("p90")
                p99 = metrics.get("p99")
                rate = metrics.get("img_per_s")
                print(
                    f"{name:<14}{B:>6}  {scope:<27}{metrics['p50']:>9.2f}"
                    f"{(f'{p90:.2f}' if p90 is not None else 'n/a'):>9}"
                    f"{(f'{p99:.2f}' if p99 is not None else 'n/a'):>9}"
                    f"{(f'{rate:.1f}' if rate is not None else 'n/a'):>10}"
                    f"{(mem if mem is not None else 0):>9.0f}"
                )
        native_rows = [
            (name, batch, stage, metric)
            for name, batch, value, _map, _mem in rows
            for stage, metric in value.get("native_stages", {}).items()
        ]
        if native_rows:
            print("native C++ stage p50 ms:")
            print(f"{'backend':<14}{'batch':>6}  {'stage':<27}{'p50':>9}")
            for name, batch, stage, metric in native_rows:
                print(f"{name:<14}{batch:>6}  {stage:<27}{metric['p50']:>9.2f}")
    if args.do_accuracy:
        print("-" * 86)
        metric_names = ("AP", "AP50", "AP75", "APs", "APm", "APl", "AR100")
        print(f"{'backend':<14}" + "".join(f"{metric:>9}" for metric in metric_names))
        for name in args.backends:
            mp = report["backends"][name].get("map")
            if mp:
                print(f"{name:<14}" + "".join(f"{_metric(mp[key]):>9}" for key in metric_names))
    if args.out:
        protected = [
            args.ann,
            args.checkpoint,
            args.onnx,
            args.engine,
            args.baseline_engine,
            args.protocol_manifest,
        ]
        if args.do_accuracy:
            protected.extend(
                Path(args.images) / coco.loadImgs(image_id)[0]["file_name"] for image_id in img_ids
            )
        elif sample_path:
            protected.append(sample_path)
        atomic_json(args.out, report, protected=protected, overwrite=args.overwrite)
        print(f"\n[profile] wrote {args.out}")
    scratch.cleanup()
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="D-FINE cross-backend profiler (latency, memory, and COCO metrics)"
    )
    p.add_argument(
        "--backends",
        nargs="+",
        default=["trt", "onnx"],
        choices=["trt", "trt-baseline", "onnx", "torch", "cpp", "cpp-graph"],
    )
    p.add_argument("--batches", type=positive_int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--warmup", type=nonnegative_int, default=20)
    p.add_argument("--iters", type=positive_int, default=100)
    p.add_argument("--rounds", type=positive_int, default=3)
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
    p.add_argument("--engine", default=os.environ.get("ENGINE", ""))
    p.add_argument("--baseline-engine", default=os.environ.get("BASELINE_ENGINE", ""))
    p.add_argument("--onnx", default=os.environ.get("ONNX", ""))
    p.add_argument(
        "--allow-onnx-cpu",
        action="store_true",
        help="allow an ONNX Runtime profile without CUDAExecutionProvider",
    )
    p.add_argument("--images", default=os.environ.get("COCO_IMAGES", ""))
    p.add_argument("--ann", default=os.environ.get("COCO_ANN", ""))
    p.add_argument(
        "--sample-image",
        default=os.environ.get("DFINE_SAMPLE_IMAGE", ""),
        help="latency input when accuracy is disabled",
    )
    p.add_argument("--out", default="")
    p.add_argument("--protocol-manifest", default="")
    p.add_argument("--overwrite", action="store_true")
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
    if a.do_accuracy and "cpp" in a.backends and a.topk > 300:
        p.error("the native runtime decodes at most 300 detections; --topk above 300 "
                "cannot be honored by the cpp backend")
    if len(set(a.batches)) != len(a.batches):
        p.error("--batches must not contain duplicates")
    if len(set(a.backends)) != len(a.backends):
        p.error("--backends must not contain duplicates")
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
        required.append(("checkpoint", "--checkpoint", "DFINE_CHECKPOINT"))
    require_arguments(p, a, required)
    if a.protocol_manifest and not Path(a.protocol_manifest).is_file():
        p.error("--protocol-manifest does not exist")
    if a.out and Path(a.out).suffix.lower() != ".json":
        p.error("--out must end in .json")
    return a


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
