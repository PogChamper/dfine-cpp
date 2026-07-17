#!/usr/bin/env python3
"""Score the C++ D-FINE detector with COCO bbox metrics.

Drives apps/dfine_coco_eval (C++): writes the image filelist, runs the binary to
produce COCO-style detections (contiguous class ids), maps those ids to category_id,
and scores with pycocotools.

The C++ side owns preprocessing + decode; this script only orchestrates + scores.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from coco_metrics import evaluate_bbox, ground_truth_summary
from eval_contract import (
    artifact_lineage_from_meta,
    byte_value,
    format_metric,
    nonnegative_int,
    normalized_model_contract,
    positive_int,
    positive_meta_int,
    probability,
    require_arguments,
    require_detections,
    resolution,
)
from evaluation_report import (
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
from pycocotools.coco import COCO

REPO = Path(__file__).resolve().parents[2]


def _remap_categories(detections: list[dict], cont2cat: dict[int, int]) -> list[dict]:
    for detection in detections:
        contiguous = int(detection.pop("category_contig"))
        detection["category_id"] = cont2cat[contiguous]
    return detections


def _execution_provenance(args) -> dict:
    full_graph = bool(args.full_graph)
    gpu_decode_option = bool(args.gpu_decode or full_graph)
    ordinary_graph_verification = (
        {
            "active": False,
            "evidence": "full-pipeline graph takes precedence for every successful evaluation call",
        }
        if full_graph
        else {
            "active": None if args.cuda_graph else False,
            "evidence": (
                "the native evaluator does not expose ordinary graph replay; runtime fallback is allowed"
                if args.cuda_graph
                else "not requested"
            ),
        }
    )
    return {
        "requested": {
            "batch": args.batch,
            "cuda_graph": args.cuda_graph,
            "filter_resolution": args.filter_res,
            "freeze": args.freeze,
            "full_graph": args.full_graph,
            "gpu_decode": args.gpu_decode,
            "letterbox": args.letterbox,
            "letterbox_pad": args.letterbox_pad,
            "letterbox_topleft": args.letterbox_topleft,
            "no_upscale": args.no_upscale,
            "own_device_memory": args.own_device_memory,
        },
        "resolved": {
            "batch": args.batch,
            "freeze_invoked": bool(args.freeze or full_graph),
            "full_pipeline_graph_option": full_graph,
            "gpu_decode_option": gpu_decode_option,
            "ordinary_cuda_graph_option": bool(args.cuda_graph),
            "dispatch_precedence": (
                "full_pipeline_graph"
                if full_graph
                else "gpu_decode_then_ordinary_graph_then_enqueue"
            ),
            "own_device_memory_option": bool(args.own_device_memory),
        },
        "verification": {
            "full_pipeline_graph": {
                "active": full_graph,
                "evidence": (
                    "successful evaluator exit requires active capture and one replay per inference call"
                    if full_graph
                    else "not requested"
                ),
            },
            "gpu_decode": {
                "active": True if full_graph else (None if args.gpu_decode else False),
                "evidence": (
                    "included in the verified full-pipeline graph"
                    if full_graph
                    else (
                        "the native evaluator permits CPU-decode fallback and does not expose activation"
                        if args.gpu_decode
                        else "not requested"
                    )
                ),
            },
            "ordinary_cuda_graph": ordinary_graph_verification,
        },
        "preprocess": {
            "resize": args.model_resize,
            "letterbox_anchor": args.model_letterbox_anchor,
            "letterbox_pad": args.model_letterbox_pad,
            "letterbox_upscale": args.model_letterbox_upscale,
        },
    }


def _read_meta(path: Path) -> dict:
    try:
        meta = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read metadata {path}: {exc}") from exc
    if not isinstance(meta, dict):
        raise ValueError(f"metadata {path} must contain a JSON object")
    return meta


def _engine_recipe(args) -> str:
    return sidecar_recipe(
        args.engine,
        "TensorRT engine",
        sidecar=args.resolved_meta,
    )


def _meta_fail(path: Path):
    def fail(message: str):
        raise ValueError(f"metadata {path}: {message}")

    return fail


def _positive_meta_int(meta: dict, key: str, path: Path) -> int | None:
    return positive_meta_int(meta, key, _meta_fail(path), optional=True)


def _normalized_model_contract(meta: dict, path: Path) -> dict:
    return normalized_model_contract(
        meta,
        _meta_fail(path),
        extra_expected={"artifact_kind": "engine", "schema_version": 1},
    )


def _engine_lineage_metadata(meta: dict, path: Path, contract: dict) -> dict:
    return artifact_lineage_from_meta(meta, "engine", contract, _meta_fail(path))


def _backend_report(args, metrics: dict) -> dict:
    engine_artifact = artifact(
        "tensorrt_engine",
        args.engine,
        recipe=_engine_recipe(args),
        runtime=package_runtime("D-FINE-cpp + TensorRT", "tensorrt"),
        sidecar=args.resolved_meta,
    )
    return {
        "artifact": engine_artifact,
        "lineage": {
            **args.engine_lineage,
            "artifact_sha256": engine_artifact["sha256"],
        },
        "model_contract": args.model_contract,
        "backend_provenance": {"execution": _execution_provenance(args)},
        "map": metrics,
    }


def _resolve_evaluation_contract(args) -> None:
    meta_path = Path(args.meta) if args.meta else discovered_sidecar(args.engine)
    if args.meta and not meta_path.is_file():
        raise ValueError(f"explicit metadata does not exist: {meta_path}")
    meta = _read_meta(meta_path) if meta_path is not None else {}
    model_contract = None
    engine_lineage = None
    if args.report:
        if meta_path is None:
            raise ValueError("--report requires a complete engine sidecar")
        model_contract = _normalized_model_contract(meta, meta_path)
        engine_lineage = _engine_lineage_metadata(meta, meta_path, model_contract)

    input_h = _positive_meta_int(meta, "input_h", meta_path) if meta_path else None
    input_w = _positive_meta_int(meta, "input_w", meta_path) if meta_path else None
    num_classes = _positive_meta_int(meta, "num_classes", meta_path) if meta_path else None
    if (input_h is None) != (input_w is None):
        raise ValueError(f"metadata {meta_path}: input_h and input_w must be specified together")
    if args.img_size is not None:
        requested = (args.img_size, args.img_size)
        if input_h is not None and requested != (input_h, input_w):
            raise ValueError(
                f"--img-size {args.img_size} contradicts metadata input "
                f"{input_h}x{input_w} in {meta_path}"
            )
        input_h, input_w = requested
    if input_h is None:
        raise ValueError(
            "input dimensions are unknown; provide an engine sidecar with input_h/input_w "
            "or pass --img-size"
        )
    if args.num_classes is not None:
        if num_classes is not None and args.num_classes != num_classes:
            raise ValueError(
                f"--num-classes {args.num_classes} contradicts metadata num_classes "
                f"{num_classes} in {meta_path}"
            )
        num_classes = args.num_classes
    if num_classes is None:
        raise ValueError(
            "class count is unknown; provide an engine sidecar with num_classes "
            "or pass --num-classes"
        )

    declared_resize = meta.get("resize", "stretch")
    declared_anchor = meta.get("letterbox_anchor", "center")
    declared_pad = meta.get("letterbox_pad", 114)
    declared_upscale = meta.get("letterbox_upscale", True)
    if declared_resize not in {"stretch", "letterbox"}:
        raise ValueError(f"metadata {meta_path}: resize must be 'stretch' or 'letterbox'")
    if declared_anchor not in {"center", "topleft"}:
        raise ValueError(f"metadata {meta_path}: letterbox_anchor must be 'center' or 'topleft'")
    if type(declared_pad) is not int or not 0 <= declared_pad <= 255:
        raise ValueError(f"metadata {meta_path}: letterbox_pad must be an integer in [0, 255]")
    if type(declared_upscale) is not bool:
        raise ValueError(f"metadata {meta_path}: letterbox_upscale must be a boolean")

    explicit_letterbox = (
        args.letterbox
        or args.letterbox_topleft
        or args.letterbox_pad is not None
        or args.no_upscale
    )
    if explicit_letterbox:
        resize = "letterbox"
        anchor = "topleft" if args.letterbox_topleft else "center"
        pad = args.letterbox_pad if args.letterbox_pad is not None else 114
        upscale = not args.no_upscale
    else:
        resize = declared_resize
        anchor = declared_anchor
        pad = declared_pad
        upscale = declared_upscale
    if args.report and resize != "stretch":
        raise ValueError("--report supports the maintained stretch preprocessing contract only")

    args.model_hw = (input_h, input_w)
    args.model_num_classes = num_classes
    args.model_resize = resize
    args.model_letterbox_anchor = anchor
    args.model_letterbox_pad = pad
    args.model_letterbox_upscale = upscale
    args.resolved_meta = str(meta_path) if meta_path is not None else None
    args.engine_lineage = engine_lineage
    args.model_contract = model_contract


def main(args):
    coco = COCO(args.ann)
    cat_ids = sorted(coco.getCatIds())
    if len(cat_ids) != args.model_num_classes:
        raise SystemExit(
            f"[cpp_coco]: annotations contain {len(cat_ids)} categories; "
            f"engine contract declares {args.model_num_classes}"
        )
    cont2cat = {i: c for i, c in enumerate(cat_ids)}  # RT-DETR/D-FINE convention
    img_ids = sorted(coco.getImgIds())
    if args.filter_res:
        # Restrict BOTH the filelist and the scored imgIds to the fixed resolution —
        # skipped images left in ev.params.imgIds would count as misses (gotcha #8).
        fw, fh = (int(v) for v in args.filter_res.split("x"))
        img_ids = [
            iid
            for iid in img_ids
            if coco.loadImgs(iid)[0]["width"] == fw and coco.loadImgs(iid)[0]["height"] == fh
        ]
    if args.limit:
        img_ids = img_ids[: args.limit]
    if not img_ids:
        raise SystemExit("[cpp_coco]: selection contains zero images")
    print(f"[cpp_coco] images={len(img_ids)} classes={len(cat_ids)}")

    scratch = tempfile.TemporaryDirectory(prefix="dfine-cpp-coco-", dir=args.tmpdir or None)
    tmpdir = Path(scratch.name)
    filelist = tmpdir / "filelist.txt"
    with open(filelist, "w") as f:
        for iid in img_ids:
            f.write(f"{iid} {coco.loadImgs(iid)[0]['file_name']}\n")

    dets_json = Path(args.out) if args.out else tmpdir / "detections.json"
    protected = [Path(args.engine), Path(args.ann), Path(args.binary)]
    if args.resolved_meta:
        protected.append(Path(args.resolved_meta))
    if args.protocol_manifest:
        protected.append(Path(args.protocol_manifest))
    protected.extend(
        Path(args.images) / coco.loadImgs(image_id)[0]["file_name"] for image_id in img_ids
    )
    for source in protected:
        if paths_alias(dets_json, source):
            raise SystemExit(f"detections output aliases input artifact: {source}")
    if args.report and paths_alias(dets_json, args.report):
        raise SystemExit("--out and --report must be different paths")
    if args.out and dets_json.exists() and not args.overwrite:
        raise SystemExit(f"detections output already exists: {dets_json}; pass --overwrite")
    cmd = [
        args.binary,
        "--engine",
        args.engine,
        "--images-dir",
        args.images,
        "--filelist",
        str(filelist),
        "--out",
        str(dets_json),
        "--threshold",
        str(args.score_thresh),
    ]
    if args.meta:
        cmd += ["--meta", args.meta]
    if args.cuda_graph:
        cmd += ["--cuda-graph"]
    if args.gpu_decode:
        cmd += ["--gpu-decode"]
    if args.own_device_memory:
        cmd += ["--own-device-memory"]
    if args.freeze:
        cmd += ["--freeze"]
    if args.full_graph:
        cmd += ["--full-graph"]
    if args.filter_res:
        cmd += ["--filter-res", args.filter_res]
    if args.batch > 1:
        cmd += ["--batch", str(args.batch)]
    if args.letterbox:
        cmd += ["--letterbox"]
    if args.letterbox_topleft:
        cmd += ["--letterbox-topleft"]
    if args.letterbox_pad is not None:
        cmd += ["--letterbox-pad", str(args.letterbox_pad)]
    if args.no_upscale:
        cmd += ["--no-upscale"]
    env = dict(os.environ)
    library_paths = [args.ld_library_path, env.get("LD_LIBRARY_PATH", "")]
    env["LD_LIBRARY_PATH"] = ":".join(path for path in library_paths if path)
    print("[cpp_coco] $", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)

    raw = json.loads(Path(dets_json).read_text())
    results = _remap_categories(raw, cont2cat)
    print(f"[cpp_coco] {len(results)} detections")
    require_detections(results, "[cpp_coco]")

    metrics = evaluate_bbox(
        coco,
        results,
        img_ids,
        model_hw=args.model_hw,
        model_resize=args.model_resize,
        letterbox_anchor=args.model_letterbox_anchor,
        letterbox_upscale=args.model_letterbox_upscale,
        letterbox_pad=args.model_letterbox_pad,
    )
    print(
        f"[cpp_coco] C++ detector AP@[.50:.95]={format_metric(metrics['AP'])}  "
        f"AP@.50={format_metric(metrics['AP50'])}  AP@.75={format_metric(metrics['AP75'])}  "
        f"AR@100={format_metric(metrics['AR100'])}"
    )
    if args.report:
        report = {
            "schema": 2,
            "images": len(img_ids),
            "ground_truth": ground_truth_summary(coco, img_ids),
            "evaluation_contract": evaluation_contract(
                coco,
                img_ids,
                args.images,
                args.ann,
                score_threshold=args.score_thresh,
                topk=300,
                inference_batch_size=args.batch,
                model_hw=args.model_hw,
                metrics_source=Path(__file__).with_name("coco_metrics.py"),
            ),
            "provenance": {
                "script_sha256": sha256_file(Path(__file__)),
                "metrics_sha256": sha256_file(Path(__file__).with_name("coco_metrics.py")),
                "runner": artifact(
                    "native_binary",
                    args.binary,
                    recipe="dfine_coco_eval",
                    runtime="native C++",
                ),
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
            "backends": {
                "cpp": _backend_report(args, metrics)
            },
        }
        atomic_json(args.report, report, protected=protected, overwrite=args.overwrite)
        print(f"[cpp_coco] wrote {args.report}")
    scratch.cleanup()
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="COCO bbox metrics for the C++ D-FINE detector")
    p.add_argument("--binary", default=str(REPO / "build" / "dfine_coco_eval"))
    p.add_argument("--engine", default=os.environ.get("ENGINE", ""))
    p.add_argument("--meta", default="")
    p.add_argument("--images", default=os.environ.get("COCO_IMAGES", ""))
    p.add_argument("--ann", default=os.environ.get("COCO_ANN", ""))
    p.add_argument("--limit", type=nonnegative_int, default=0, help="0 = all val images")
    p.add_argument("--score-thresh", type=probability, default=0.001)
    p.add_argument("--num-classes", type=positive_int, default=None)
    p.add_argument(
        "--img-size",
        type=positive_int,
        default=None,
        help="square engine input; inferred from metadata when omitted",
    )
    p.add_argument("--out", default="")
    p.add_argument("--report", default="", help="write metrics JSON")
    p.add_argument("--protocol-manifest", default="")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--tmpdir", default="")
    p.add_argument(
        "--ld-library-path", default="", help="prepend this directory to LD_LIBRARY_PATH"
    )
    p.add_argument("--cuda-graph", action="store_true", help="pass --cuda-graph to the binary")
    p.add_argument("--gpu-decode", action="store_true", help="decode engine outputs on the GPU")
    p.add_argument("--own-device-memory", action="store_true", help="pass --own-device-memory")
    p.add_argument("--freeze", action="store_true", help="pass --freeze (frozen-memory contract)")
    p.add_argument(
        "--full-graph",
        action="store_true",
        help="use the full-pipeline graph; pair with --filter-res",
    )
    p.add_argument(
        "--filter-res",
        type=resolution,
        default=None,
        help="WxH: eval only images of exactly this size (fixed-resolution regime)",
    )
    p.add_argument("--batch", type=positive_int, default=1, help="pass --batch to the binary")
    p.add_argument(
        "--letterbox",
        action="store_true",
        help="letterbox preprocessing (validated against letterbox_eval.py hosts)",
    )
    p.add_argument("--letterbox-topleft", action="store_true")
    p.add_argument("--letterbox-pad", type=byte_value, default=None)
    p.add_argument("--no-upscale", action="store_true")
    args = p.parse_args(argv)
    require_arguments(
        p,
        args,
        [
            ("engine", "--engine", "ENGINE"),
            ("images", "--images", "COCO_IMAGES"),
            ("ann", "--ann", "COCO_ANN"),
        ],
    )
    if args.full_graph and not args.filter_res:
        p.error("--full-graph requires --filter-res WxH")
    if args.protocol_manifest and not Path(args.protocol_manifest).is_file():
        p.error("--protocol-manifest does not exist")
    for value, label in ((args.out, "--out"), (args.report, "--report")):
        if value and Path(value).suffix.lower() != ".json":
            p.error(f"{label} must end in .json")
    try:
        _resolve_evaluation_contract(args)
    except ValueError as exc:
        p.error(str(exc))
    return args


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
