#!/usr/bin/env python3
"""COCO bbox evaluation for the raw D-FINE engine and reference backends.

Measures the residual TensorRT-vs-PyTorch gap by running both backends over COCO
val and scoring with pycocotools. The decode is the native runtime reference:

    sigmoid(logits) -> top-300 over (query x class) -> label=idx%C, query=idx//C
    -> cxcywh(normalized) to xyxy -> scale by original (W,H)  [stretch preprocessing]

Class ids are mapped contiguous 0..C-1 -> COCO category_id via the sorted category ids
(the RT-DETR/D-FINE convention).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# profile.py in this dir shadows stdlib `profile` (imported by cProfile via
# torchvision->torch._dynamo when the torch backend loads). Move the scripts dir to the
# END of sys.path so stdlib wins the name but sibling modules (cuda_env) still import.
_scripts_dir = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _scripts_dir)]
sys.path.append(_scripts_dir)
_trt_files_dir = str(Path(__file__).resolve().parents[1])
sys.path[:] = [path for path in sys.path if path != _trt_files_dir]
sys.path.insert(0, _trt_files_dir)

import cv2
import numpy as np
import tensorrt as trt
import torch
from coco_metrics import evaluate_bbox, ground_truth_summary
from eval_contract import (
    STRETCH_PREPROCESS,
    artifact_lineage_from_meta,
    format_metric,
    nonnegative_int,
    normalized_model_contract,
    positive_int,
    probability,
    require_arguments,
    require_complete_images,
    require_detection_outputs,
    require_detections,
    require_trt_success,
)
from evaluation_report import (
    artifact,
    atomic_json,
    discovered_sidecar,
    environment_metadata,
    evaluation_contract,
    package_runtime,
    sha256_file,
    sidecar_recipe,
)
from pycocotools.coco import COCO


def _sidecar_fail(sidecar: Path):
    def fail(message: str):
        raise SystemExit(f"[coco]: {sidecar}: {message}")

    return fail


def _normalized_model_contract(meta: dict, sidecar: Path) -> dict:
    return normalized_model_contract(meta, _sidecar_fail(sidecar))


def _artifact_contract(path: str | Path, kind: str) -> tuple[dict, dict, Path]:
    sidecar = discovered_sidecar(path)
    if sidecar is None:
        raise SystemExit(f"[coco]: {kind} artifact requires an adjacent JSON sidecar: {path}")
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"[coco]: cannot read {kind} sidecar {sidecar}: {exc}") from exc
    if not isinstance(meta, dict):
        raise SystemExit(f"[coco]: {sidecar}: sidecar must contain a JSON object")
    artifact_kind = meta.get("artifact_kind")
    if artifact_kind not in (None, kind):
        raise SystemExit(
            f"[coco]: {sidecar}: artifact_kind {artifact_kind!r} does not describe {kind}"
        )
    return _normalized_model_contract(meta, sidecar), meta, sidecar


def _artifact_lineage(
    path: str | Path,
    kind: str,
    meta: dict,
    sidecar: Path,
    contract: dict,
) -> dict:
    lineage = artifact_lineage_from_meta(meta, kind, contract, _sidecar_fail(sidecar))
    lineage["artifact_sha256"] = sha256_file(path)
    return lineage


def _torch_model_contract(args, model, checkpoint_sha256: str) -> dict:
    initial_queries = int(model.decoder.num_queries)
    cascade = getattr(args, "cascade", None)
    output_queries = initial_queries
    if cascade:
        try:
            _layer, output_queries = (int(value) for value in cascade.split(":"))
        except ValueError:
            raise SystemExit(f"[coco]: invalid cascade specification {cascade!r}") from None
    eval_idx = int(model.decoder.eval_idx)
    return {
        "model": "d-fine",
        "variant": args.model_name,
        "task": "detect",
        "input_h": args.img_size,
        "input_w": args.img_size,
        "num_classes": args.num_classes,
        "initial_queries": initial_queries,
        "num_queries": output_queries,
        "eval_idx": eval_idx,
        "cascade": cascade,
        "checkpoint_sha256": checkpoint_sha256,
        "preprocess": dict(STRETCH_PREPROCESS),
    }


def _require_matching_model_contracts(contracts: dict[str, dict]) -> None:
    reference_name, reference = next(iter(contracts.items()))
    for name, contract in contracts.items():
        for field in reference:
            if contract.get(field) != reference[field]:
                raise SystemExit(
                    f"[coco]: {name} model contract differs from {reference_name}: "
                    f"{field}={contract.get(field)!r}, expected {reference[field]!r}"
                )


def _require_evaluation_arguments(contract: dict, args, backend: str) -> None:
    expected = {
        "variant": args.model_name,
        "input_h": args.img_size,
        "input_w": args.img_size,
        "num_classes": args.num_classes,
    }
    for field, value in expected.items():
        if contract[field] != value:
            raise SystemExit(
                f"[coco]: {backend} sidecar {field}={contract[field]!r}; "
                f"evaluation requested {value!r}"
            )


def _require_query_count(logits, expected: int, backend: str) -> None:
    if logits.shape[1] != expected:
        raise SystemExit(
            f"[coco]: {backend} returned {logits.shape[1]} queries; "
            f"artifact contract requires {expected}"
        )


def _require_engine_source(lineage: dict, onnx_path: str | Path) -> None:
    if lineage["onnx_sha256"] != sha256_file(onnx_path):
        raise SystemExit(
            "[coco]: engine sidecar onnx_sha256 does not match the selected ONNX artifact"
        )


def configure_torch_numeric_policy() -> dict:
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }


def preprocess(bgr: np.ndarray, img: int) -> torch.Tensor:
    resized = cv2.resize(bgr, (img, img), interpolation=cv2.INTER_LINEAR)  # stretch
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return torch.from_numpy(chw).unsqueeze(0).contiguous()


def decode(
    logits: np.ndarray, boxes: np.ndarray, orig_w: int, orig_h: int, num_classes: int, topk: int
):
    """Convert raw outputs to pixel boxes, class indices, and scores."""
    prob = 1.0 / (1.0 + np.exp(-logits[0]))  # [Q, C]
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
    b = boxes[0, q]  # [topk, 4] cxcywh normalized
    cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    x1 = (cx - 0.5 * w) * orig_w
    y1 = (cy - 0.5 * h) * orig_h
    bw = w * orig_w
    bh = h * orig_h
    return np.stack([x1, y1, bw, bh], axis=1), labels, scores


class TorchBackend:
    def __init__(self, args):
        import export_dfine_onnx as exporter
        from dfine_model import build_model

        numeric_policy = configure_torch_numeric_policy()
        model = build_model(args.model_name, args.num_classes, (args.img_size, args.img_size))
        report = exporter.load_checkpoint_state(
            model,
            Path(args.checkpoint),
            allow_partial=False,
            allow_unsafe=False,
        )
        if report["mode"] != "strict":
            raise RuntimeError("PyTorch reference requires a strict checkpoint load")
        exporter.apply_sliders(model, args)
        self.model_contract = _torch_model_contract(args, model, report["sha256"])
        self.lineage = {
            "artifact_kind": "checkpoint",
            "precision_mode": "fp32",
            "checkpoint_sha256": report["sha256"],
            "artifact_sha256": report["sha256"],
        }
        model_root = Path(exporter.__file__).resolve().parents[1] / "dfine_model"
        self.provenance = {
            "checkpoint_load": report,
            "numeric_policy": numeric_policy,
            "sliders": {
                "num_queries": args.num_queries,
                "eval_idx": args.eval_idx,
                "cascade": args.cascade,
            },
            "bundled_model": {
                "path": str(model_root),
                "manifest": exporter._model_source_manifest(model_root),
                "validated_upstream_commit": exporter._validated_source_revision(),
            },
            "exporter": {
                "path": str(Path(exporter.__file__).resolve()),
                "sha256": exporter._exporter_sha256(),
            },
        }
        self.m = model.cuda().deploy().eval()

    @torch.no_grad()
    def __call__(self, x):
        o = self.m(x.cuda())
        return o["pred_logits"].float().cpu().numpy(), o["pred_boxes"].float().cpu().numpy()


class OrtBackend:
    def __init__(self, onnx_path, *, allow_cpu: bool = False):
        import cuda_env  # bootstraps onnxruntime-gpu (WSL libcuda path + preload_dlls)

        ort = cuda_env.bootstrap()
        self.sess, _ = cuda_env.make_session(onnx_path, use_tf32=False)
        providers = self.sess.get_providers()
        if "CUDAExecutionProvider" not in providers and not allow_cpu:
            raise RuntimeError(
                "ONNX Runtime did not activate CUDAExecutionProvider; "
                "pass --allow-onnx-cpu only for an explicitly CPU-backed comparison"
            )
        provider_options = self.sess.get_provider_options()
        cuda_options = provider_options.get("CUDAExecutionProvider", {})
        if "CUDAExecutionProvider" in providers and str(cuda_options.get("use_tf32")) != "0":
            raise RuntimeError("ONNX Runtime CUDAExecutionProvider did not disable TF32")
        self.provenance = {
            "runtime_version": ort.__version__,
            "runtime_device": ort.get_device(),
            "providers": providers,
            "provider_options": provider_options,
            "cuda_required": not allow_cpu,
            "numeric_policy": {"tf32_requested": False},
        }
        print(f"[coco] onnx providers: {providers}")

    def __call__(self, x):
        o = self.sess.run(["logits", "boxes"], {"images": x.numpy()})
        return o[0], o[1]


def _validated_engine_names(engine) -> list[str]:
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    inputs = [name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]
    if inputs != ["images"]:
        raise RuntimeError("engine must expose exactly one input named 'images'")
    if engine.get_tensor_dtype("images") != trt.DataType.FLOAT:
        raise RuntimeError("engine input 'images' must be FP32")
    if "logits" not in names or "boxes" not in names:
        raise RuntimeError("engine must expose outputs named 'logits' and 'boxes'")
    return names


class EngineBackend:
    def __init__(self, engine_path):
        runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"TensorRT could not deserialize engine: {engine_path}")
        self.ctx = self.engine.create_execution_context()
        if self.ctx is None:
            raise RuntimeError("TensorRT could not create an execution context")
        self.names = _validated_engine_names(self.engine)
        self.stream = torch.cuda.Stream()
        self._cached_call = None

    def prepare(self, x):
        require_trt_success(self.ctx.set_input_shape("images", tuple(x.shape)), "input shape")
        if x.device.type != "cpu" or x.dtype != torch.float32 or not x.is_contiguous():
            raise ValueError("TensorRT input must be a contiguous CPU FP32 tensor")

        host_input = torch.empty_like(x, pin_memory=True)
        host_input.copy_(x)
        device_input = torch.empty_like(host_input, device="cuda")
        dtype_map = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
        }
        device_outputs = {}
        host_outputs = {}
        for n in self.names:
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT:
                trt_dtype = self.engine.get_tensor_dtype(n)
                if trt_dtype not in dtype_map:
                    raise RuntimeError(f"unsupported output dtype for {n}: {trt_dtype}")
                shape = tuple(self.ctx.get_tensor_shape(n))
                if not shape or any(dim <= 0 for dim in shape):
                    raise RuntimeError(f"unresolved output shape for {n}: {shape}")
                device = torch.empty(shape, dtype=dtype_map[trt_dtype], device="cuda")
                host = torch.empty(shape, dtype=dtype_map[trt_dtype], pin_memory=True)
                device_outputs[n] = device
                host_outputs[n] = host
        call = _PreparedEngineCall(
            self.ctx,
            self.stream,
            host_input,
            device_input,
            host_outputs,
            device_outputs,
        )
        call.bind()
        return call

    def reusable_call(self, x):
        cached = getattr(self, "_cached_call", None)
        if cached is None or tuple(cached.host_input.shape) != tuple(x.shape):
            self._cached_call = self.prepare(x)
        else:
            cached.set_host_input(x)
            cached.bind()
        return self._cached_call

    def __call__(self, x):
        call = self.reusable_call(x)
        call.transfer()
        call.synchronize()
        call.require_finite(device=True, host=True)
        return call.numpy_outputs()


class _PreparedEngineCall:
    def __init__(
        self,
        context,
        stream,
        host_input,
        device_input,
        host_outputs,
        device_outputs,
    ):
        self.context = context
        self.stream = stream
        self.host_input = host_input
        self.device_input = device_input
        self.host_outputs = host_outputs
        self.device_outputs = device_outputs

    def bind(self):
        require_trt_success(
            self.context.set_input_shape("images", tuple(self.device_input.shape)), "input shape"
        )
        require_trt_success(
            self.context.set_tensor_address("images", self.device_input.data_ptr()), "input address"
        )
        for name, device in self.device_outputs.items():
            require_trt_success(
                self.context.set_tensor_address(name, device.data_ptr()),
                f"output address for {name}",
            )

    def set_host_input(self, value):
        if value.device.type != "cpu" or value.dtype != torch.float32 or not value.is_contiguous():
            raise ValueError("TensorRT input must be a contiguous CPU FP32 tensor")
        self.host_input.copy_(value)

    def upload(self):
        with torch.cuda.stream(self.stream):
            self.device_input.copy_(self.host_input, non_blocking=True)

    def enqueue(self):
        require_trt_success(self.context.execute_async_v3(self.stream.cuda_stream), "enqueueV3")

    def transfer(self):
        with torch.cuda.stream(self.stream):
            self.device_input.copy_(self.host_input, non_blocking=True)
            self.enqueue()
            for name, device in self.device_outputs.items():
                self.host_outputs[name].copy_(device, non_blocking=True)

    def synchronize(self):
        self.stream.synchronize()

    def require_finite(self, *, device: bool, host: bool):
        tensors = []
        if device:
            tensors.extend(self.device_outputs.values())
        if host:
            tensors.extend(self.host_outputs.values())
        if not all(torch.isfinite(tensor).all().item() for tensor in tensors):
            raise RuntimeError("TensorRT outputs contain NaN or Inf")

    def numpy_outputs(self):
        return (
            self.host_outputs["logits"].float().numpy(),
            self.host_outputs["boxes"].float().numpy(),
        )


def evaluate(
    coco: COCO,
    dets: list,
    label: str,
    img_ids: list,
    model_hw: tuple[int, int] | None = None,
) -> dict:
    require_detections(dets, f"[coco] {label}")
    metrics = evaluate_bbox(coco, dets, img_ids, model_hw=model_hw)
    print(
        f"[coco] {label}: AP@[.50:.95]={format_metric(metrics['AP'])}  "
        f"AP@.50={format_metric(metrics['AP50'])}  AP@.75={format_metric(metrics['AP75'])}  "
        f"AR@100={format_metric(metrics['AR100'])}"
    )
    return metrics


def backend_artifact(name: str, args) -> dict:
    if name == "torch":
        recipe = [
            "eager-fp32",
            f"model={args.model_name}",
            f"queries={getattr(args, 'num_queries', None) or 300}",
        ]
        if getattr(args, "eval_idx", None) is not None:
            recipe.append(f"eval_idx={args.eval_idx}")
        if getattr(args, "cascade", None):
            recipe.append(f"cascade={args.cascade}")
        return artifact(
            "checkpoint",
            args.checkpoint,
            recipe=";".join(recipe),
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
    if name == "engine":
        path = args.engine
        return artifact(
            "tensorrt_engine",
            path,
            recipe=sidecar_recipe(path, "TensorRT engine"),
            runtime=package_runtime("TensorRT", "tensorrt"),
            sidecar=discovered_sidecar(path),
        )
    raise ValueError(f"unknown backend: {name}")


def main(args):
    coco = COCO(args.ann)
    cat_ids = sorted(coco.getCatIds())
    if len(cat_ids) != args.num_classes:
        raise SystemExit(
            f"[coco]: annotations contain {len(cat_ids)} categories; "
            f"--num-classes is {args.num_classes}"
        )
    cont2cat = {i: c for i, c in enumerate(cat_ids)}  # RT-DETR/D-FINE convention
    img_ids = sorted(coco.getImgIds())
    if args.limit:
        img_ids = img_ids[: args.limit]
    if not img_ids:
        raise SystemExit("[coco]: selection contains zero images")
    print(f"[coco] images={len(img_ids)} classes={len(cat_ids)} backends={args.backends}")

    backends = {}
    contracts = {}
    lineages = {}
    if "torch" in args.backends:
        backends["torch"] = TorchBackend(args)
        contracts["torch"] = backends["torch"].model_contract
        lineages["torch"] = backends["torch"].lineage
    if "onnx" in args.backends:
        contract, metadata, sidecar = _artifact_contract(args.onnx, "onnx")
        _require_evaluation_arguments(contract, args, "onnx")
        backends["onnx"] = OrtBackend(args.onnx, allow_cpu=getattr(args, "allow_onnx_cpu", False))
        contracts["onnx"] = contract
        lineages["onnx"] = _artifact_lineage(
            args.onnx, "onnx", metadata, sidecar, contract
        )
    if "engine" in args.backends:
        contract, metadata, sidecar = _artifact_contract(args.engine, "engine")
        _require_evaluation_arguments(contract, args, "engine")
        backends["engine"] = EngineBackend(args.engine)
        contracts["engine"] = contract
        lineages["engine"] = _artifact_lineage(
            args.engine, "engine", metadata, sidecar, contract
        )

    _require_matching_model_contracts(contracts)
    if "onnx" in backends and "engine" in backends:
        _require_engine_source(lineages["engine"], args.onnx)

    out = {k: [] for k in backends}
    processed = []
    img_dir = Path(args.images)
    for n, iid in enumerate(img_ids):
        info = coco.loadImgs(iid)[0]
        image_path = img_dir / info["file_name"]
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"[coco]: cannot read image: {image_path}")
        processed.append(iid)
        H, W = bgr.shape[:2]
        x = preprocess(bgr, args.img_size)
        for name, be in backends.items():
            log, box = be(x)
            log, box = require_detection_outputs(log, box, 1, args.num_classes, f"[coco] {name}")
            _require_query_count(log, contracts[name]["num_queries"], name)
            boxes, labels, scores = decode(log, box, W, H, args.num_classes, args.topk)
            for bb, lb, sc in zip(boxes, labels, scores):
                if sc < args.score_thresh:
                    continue
                out[name].append(
                    {
                        "image_id": iid,
                        "category_id": cont2cat[int(lb)],
                        "bbox": [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])],
                        "score": float(sc),
                    }
                )
        if (n + 1) % 200 == 0:
            print(f"[coco]   processed {n + 1}/{len(img_ids)}")

    require_complete_images(len(img_ids), len(processed), "[coco]")
    report = {
        "schema": 2,
        "images": len(processed),
        "ground_truth": ground_truth_summary(coco, processed),
        "evaluation_contract": evaluation_contract(
            coco,
            processed,
            args.images,
            args.ann,
            score_threshold=args.score_thresh,
            topk=args.topk,
            inference_batch_size=1,
            model_hw=(args.img_size, args.img_size),
            metrics_source=Path(__file__).with_name("coco_metrics.py"),
        ),
        "provenance": {
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
            "environment": environment_metadata(),
        },
        "backends": {},
    }
    for name in backends:
        entry = {
            "artifact": backend_artifact(name, args),
            "lineage": lineages[name],
            "model_contract": contracts[name],
            "map": evaluate(
                coco,
                out[name],
                name,
                processed,
                model_hw=(args.img_size, args.img_size),
            ),
        }
        backend_provenance = getattr(backends[name], "provenance", None)
        if backend_provenance is not None:
            entry["backend_provenance"] = backend_provenance
        report["backends"][name] = entry
    if args.report:
        protected = [args.ann, args.checkpoint, args.onnx, args.engine, args.protocol_manifest]
        protected.extend(
            Path(args.images) / coco.loadImgs(image_id)[0]["file_name"] for image_id in processed
        )
        atomic_json(args.report, report, protected=protected, overwrite=args.overwrite)
        print(f"[coco] wrote {args.report}")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="COCO bbox metrics for D-FINE backends")
    p.add_argument("--model-name", default="m")
    p.add_argument("--num-classes", type=positive_int, default=80)
    p.add_argument("--img-size", type=positive_int, default=640)
    p.add_argument("--topk", type=positive_int, default=300)
    p.add_argument("--score-thresh", type=probability, default=0.001)
    p.add_argument("--num-queries", type=positive_int, default=None)
    p.add_argument("--eval-idx", type=int, default=None)
    p.add_argument("--cascade", default=None, metavar="K:KEEP")
    p.add_argument("--limit", type=nonnegative_int, default=0, help="0 = all val images")
    p.add_argument(
        "--backends",
        nargs="+",
        default=["engine", "torch"],
        choices=["engine", "torch", "onnx"],
    )
    p.add_argument("--checkpoint", default=os.environ.get("DFINE_CHECKPOINT", ""))
    p.add_argument("--engine", default=os.environ.get("ENGINE", ""))
    p.add_argument("--onnx", default=os.environ.get("ONNX", ""))
    p.add_argument(
        "--allow-onnx-cpu",
        action="store_true",
        help="allow an ONNX Runtime comparison without CUDAExecutionProvider",
    )
    p.add_argument("--images", default=os.environ.get("COCO_IMAGES", ""))
    p.add_argument("--ann", default=os.environ.get("COCO_ANN", ""))
    p.add_argument("--report", default="", help="write metrics JSON")
    p.add_argument("--protocol-manifest", default="")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    if len(set(args.backends)) != len(args.backends):
        p.error("--backends must not contain duplicates")
    required = [("images", "--images", "COCO_IMAGES"), ("ann", "--ann", "COCO_ANN")]
    if "engine" in args.backends:
        required.append(("engine", "--engine", "ENGINE"))
    if "onnx" in args.backends:
        required.append(("onnx", "--onnx", "ONNX"))
    if "torch" in args.backends:
        required.append(("checkpoint", "--checkpoint", "DFINE_CHECKPOINT"))
    elif any((args.num_queries is not None, args.eval_idx is not None, bool(args.cascade))):
        p.error("--num-queries, --eval-idx, and --cascade apply only to the torch backend")
    require_arguments(p, args, required)
    if args.protocol_manifest and not Path(args.protocol_manifest).is_file():
        p.error("--protocol-manifest does not exist")
    if args.report and Path(args.report).suffix.lower() != ".json":
        p.error("--report must end in .json")
    return args


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
