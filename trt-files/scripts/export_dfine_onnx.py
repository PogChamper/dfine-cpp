#!/usr/bin/env python3
"""Export a D-FINE detection model to a RAW two-output ONNX graph for D-FINE-cpp.

The exported graph takes a single ``images`` input and returns the decoder's raw
``logits`` (pre-sigmoid, [N, num_queries, num_classes]) and ``boxes`` (normalized
cxcywh, [N, num_queries, 4]). Sigmoid + top-k + box conversion are intentionally
left out of the graph and performed in C++ (see docs/synthesis/01_PLAN §5, §8).

The FDR/Integral/LQE box decode and the deformable attention stay inside the graph
regardless of this choice; ``model.deploy()`` folds the weighting vector and truncates
the decoder to its evaluation layers.

A JSON sidecar with the same stem describes the engine contract (input geometry,
normalization, per-variant constants) so the C++ runtime stays model-generic.

Model construction uses the D-FINE-seg package, whose detection subnetwork is
architecturally identical to authorial D-FINE (verified: docs/research/V00 §V07);
the resulting graph is representative of either repo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# This script lives in trt-files/scripts alongside profile.py, whose name shadows the
# stdlib `profile` module that cProfile (pulled in by torchvision -> torch._dynamo)
# imports. Drop the scripts dir from the front of sys.path so stdlib wins; this script
# imports no sibling module, so removing it is safe. (Not an issue pre-M1 when there was
# no profile.py to shadow.)
_scripts_dir = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", _scripts_dir)]

import torch
import torch.nn as nn


def _add_repo_to_path(repo_root: Path) -> None:
    """Make the D-FINE-seg ``src`` package importable."""
    root = str(repo_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


class RawDetect(nn.Module):
    """Wrap a D-FINE model so its forward returns ``(logits, boxes)`` tensors."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor):
        out = self.model(images)
        return out["pred_logits"], out["pred_boxes"]


# --- Explicit gather-bilinear deformable-attention core -----------------------------
# Replaces F.grid_sample in the deformable cross-attention with an explicit
# gather-bilinear of grid_sample(bilinear, zeros, align_corners=False). Same math, but
# expressed as Gather + arithmetic, which TensorRT executes in exact FP32. The native
# GridSample node is bit-exact in isolation but is compiled divergently IN CONTEXT by
# TRT, costing ~10 AP on D-FINE's FDR decode (docs/impl/M0_STATUS.md). This is the fix,
# and it needs no TensorRT plugin.

def _bilinear_gather(value_l, grid_l, h, w):
    M, c = value_l.shape[0], value_l.shape[1]
    Lq, P = grid_l.shape[1], grid_l.shape[2]
    gx, gy = grid_l[..., 0], grid_l[..., 1]
    ix = (gx + 1) * w / 2 - 0.5   # align_corners=False unnormalize
    iy = (gy + 1) * h / 2 - 0.5
    x0 = torch.floor(ix); y0 = torch.floor(iy)
    x1 = x0 + 1; y1 = y0 + 1
    wx1 = ix - x0; wx0 = 1 - wx1
    wy1 = iy - y0; wy0 = 1 - wy1
    vflat = value_l.reshape(M, c, h * w)

    def _clip(t, hi):
        # min/max instead of .clamp(): the dynamo exporter (opset>=18) lowers .clamp() to a Clip
        # whose constant min/max inputs TensorRT 10.13's parser rejects ("input was not registered").
        # minimum/maximum lower to Min/Max, which parse cleanly. (legacy tracer is unaffected.)
        return torch.minimum(torch.maximum(t, t.new_zeros(())), t.new_full((), float(hi)))

    def corner(xc, yc, wgt):
        valid = ((xc >= 0) & (xc <= w - 1) & (yc >= 0) & (yc <= h - 1)).to(value_l.dtype)
        idx = (_clip(yc, h - 1) * w + _clip(xc, w - 1)).long().reshape(M, 1, Lq * P).expand(M, c, Lq * P)
        return torch.gather(vflat, 2, idx).reshape(M, c, Lq, P) * (wgt * valid).unsqueeze(1)

    return (corner(x0, y0, wx0 * wy0) + corner(x1, y0, wx1 * wy0)
            + corner(x0, y1, wx0 * wy1) + corner(x1, y1, wx1 * wy1))


def _explicit_deformable_core(value, value_spatial_shapes, sampling_locations,
                              attention_weights, num_points_list, method="default"):
    bs, n_head, c, _ = value[0].shape
    _, Len_q, _, _, _ = sampling_locations.shape
    grids = (2 * sampling_locations - 1).permute(0, 2, 1, 3, 4).flatten(0, 1)
    grids_list = grids.split(num_points_list, dim=-2)
    sampled = [_bilinear_gather(value[lvl].reshape(bs * n_head, c, int(h), int(w)), grids_list[lvl], int(h), int(w))
               for lvl, (h, w) in enumerate(value_spatial_shapes)]
    attn = attention_weights.permute(0, 2, 1, 3).reshape(bs * n_head, 1, Len_q, sum(num_points_list))
    out = (torch.concat(sampled, dim=-1) * attn).sum(-1).reshape(bs, n_head * c, Len_q)
    return out.permute(0, 2, 1)


def patch_explicit_deform(model: nn.Module) -> int:
    import functools
    n = 0
    for layer in model.decoder.decoder.layers:
        layer.cross_attn.ms_deformable_attn_core = functools.partial(_explicit_deformable_core, method="default")
        n += 1
    return n


def _scalar(value) -> float:
    t = torch.as_tensor(value).reshape(-1)
    return float(t[0])


# COCO-80 display names in contiguous-id order (matches include/dfine/core/coco_classes.hpp).
COCO80_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def _resolve_class_names(args: argparse.Namespace) -> list[str]:
    """--class-names as a file (one per line) or a comma list; COCO-80 by default
    for 80-class models; empty (field omitted) otherwise."""
    if args.class_names:
        p = Path(args.class_names)
        names = ([ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
                 if p.is_file() else [s.strip() for s in args.class_names.split(",") if s.strip()])
        if len(names) != args.num_classes:
            raise SystemExit(f"--class-names gave {len(names)} names for "
                             f"{args.num_classes} classes")
        return names
    return COCO80_NAMES if args.num_classes == 80 else []


def _collect_meta(model: nn.Module, args: argparse.Namespace) -> dict:
    """Read engine-contract constants off the built (deployed) model."""
    dec = model.decoder
    enc = model.encoder
    eval_idx = int(getattr(dec, "eval_idx"))
    class_names = _resolve_class_names(args)
    return {
        "model": "d-fine",
        "variant": args.model_name,
        "task": "detect",
        "input_h": args.img_size,
        "input_w": args.img_size,
        "num_classes": args.num_classes,
        "num_queries": int(dec.num_queries),
        "reg_max": int(dec.reg_max),
        "reg_scale": round(_scalar(dec.reg_scale), 6),
        "num_decoder_layers": len(dec.decoder.layers),
        "eval_idx": eval_idx,
        "num_levels": int(dec.num_levels),
        "hidden_dim": int(dec.hidden_dim),
        "feat_strides": list(getattr(enc, "feat_strides", [])),
        "input_names": ["images"],
        "output_names": ["logits", "boxes"],
        "logits_shape": ["N", int(dec.num_queries), args.num_classes],
        "boxes_shape": ["N", int(dec.num_queries), 4],
        "box_format": "cxcywh_normalized",
        "score_activation": "sigmoid",
        "color_order": "RGB",
        "channel_layout": "NCHW",
        "normalize": "div255",
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "resize": "stretch",
        "nms": "none",
        "has_masks": False,
        "dynamic_batch": True,
        "max_batch": args.max_batch,
        "opset": args.opset,
        "deform_core": args.deform,
        "trt_min_version": "8.5",
        **({"class_names": class_names} if class_names else {}),
    }


def build_detection_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    from src.d_fine.dfine import build_model  # noqa: E402  (path set at runtime)
    from src.d_fine.utils import load_tuning_state  # noqa: E402

    model = build_model(
        model_name=args.model_name,
        num_classes=args.num_classes,
        enable_mask_head=False,
        device=device,
        img_size=(args.img_size, args.img_size),
        in_channels=3,
        pretrained_model_path=None,
        pretrained_backbone=False,
    )
    ckpt = Path(args.checkpoint).resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    model = load_tuning_state(model, str(ckpt))
    return model.to(device)


def export(args: argparse.Namespace) -> None:
    _add_repo_to_path(Path(args.dfine_src))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[export] device={device} variant={args.model_name} classes={args.num_classes}")

    model = build_detection_model(args, device)
    model.deploy()  # eval() + fold weighting vector + truncate to eval layers
    model.eval()

    if args.deform == "explicit":
        n = patch_explicit_deform(model)
        print(f"[export] patched {n} deformable cores -> explicit gather-bilinear (TRT-accurate, no GridSample)")

    # Trace with batch >= 2 so the tracer cannot constant-fold the batch axis to a
    # literal 1. D-FINE generates anchors as [1, sum_hw, 4]; with a batch-1 dummy the
    # query-selection GatherElements bakes a data extent of 1 and the engine rejects
    # any N>1. A batch-2 trace keeps the axis symbolic across the whole graph.
    dummy = torch.randn(args.trace_batch, 3, args.img_size, args.img_size, device=device)
    with torch.no_grad():
        out = model(dummy)
    if not (isinstance(out, dict) and "pred_logits" in out and "pred_boxes" in out):
        raise RuntimeError(f"unexpected eval output keys: {list(out) if isinstance(out, dict) else type(out)}")
    print(f"[export] eval forward ok: logits={tuple(out['pred_logits'].shape)} boxes={tuple(out['pred_boxes'].shape)}")

    meta = _collect_meta(model, args)
    print(f"[export] meta: {json.dumps({k: meta[k] for k in ('variant','num_queries','reg_max','reg_scale','num_decoder_layers','eval_idx','num_levels','hidden_dim','feat_strides')})}")

    onnx_path = Path(args.output).resolve()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = {"images": {0: "N"}, "logits": {0: "N"}, "boxes": {0: "N"}}
    export_kwargs = dict(
        input_names=["images"],
        output_names=["logits", "boxes"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
    )
    wrapped = RawDetect(model)
    try:
        torch.onnx.export(wrapped, (dummy,), str(onnx_path), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(wrapped, (dummy,), str(onnx_path), **export_kwargs)
    print(f"[export] wrote {onnx_path}")

    if not args.no_simplify:
        _simplify(onnx_path, args)

    _verify_graph(onnx_path, meta)

    sidecar = onnx_path.with_suffix(".json")
    sidecar.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[export] wrote sidecar {sidecar}")


def _simplify(onnx_path: Path, args: argparse.Namespace) -> None:
    """Best-effort onnxsim pass. The onnxsim API differs across versions and can
    specialize a dynamic axis; any failure leaves the un-simplified graph in place,
    and _verify_graph re-checks that the batch axis stayed symbolic."""
    try:
        import onnx
        from onnxsim import simplify
    except Exception as exc:  # noqa: BLE001
        print(f"[export] onnxsim unavailable ({exc}); skipping")
        return
    model = onnx.load(str(onnx_path))
    try:
        simplified, ok = simplify(model)
    except Exception as exc:  # noqa: BLE001
        print(f"[export] onnxsim raised ({exc}); keeping unsimplified graph")
        return
    if not ok:
        print("[export] onnxsim validation returned False; keeping unsimplified graph")
        return
    onnx.save(simplified, str(onnx_path))
    print("[export] onnxsim simplified")


def _verify_graph(onnx_path: Path, meta: dict) -> None:
    import onnx

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    inferred = onnx.shape_inference.infer_shapes(model)
    graph = inferred.graph

    grid = [n for n in graph.node if n.op_type == "GridSample"]
    n_grid = len(grid)
    if meta.get("deform_core") == "explicit":
        # The explicit gather-bilinear core (the TRT-accurate fix) leaves NO GridSample
        # node — the deformable sampling is Gather + arithmetic.
        n_gather = sum(1 for n in graph.node if n.op_type in ("Gather", "GatherElements", "GatherND"))
        print(f"[verify] deform_core=explicit: GridSample={n_grid} (expect 0), Gather={n_gather}")
        if n_grid != 0:
            raise AssertionError(f"explicit core must have 0 GridSample nodes, found {n_grid}")
        meta["gridsample_nodes"] = 0
        _verify_io(graph, meta)
        return

    # Native GridSample core: one 4D GridSample per feature level per decoder layer
    # (the legacy tracer unrolls the per-level loop and the eval-truncated layer stack).
    expected = meta["num_levels"] * meta["num_decoder_layers"]
    print(f"[verify] GridSample nodes: {n_grid} (expected num_levels*num_layers = "
          f"{meta['num_levels']}*{meta['num_decoder_layers']} = {expected})")
    if n_grid != expected:
        raise AssertionError(f"expected {expected} GridSample nodes, found {n_grid}")

    ranks = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        ranks[vi.name] = len(vi.type.tensor_type.shape.dim)
    bad = []
    for n in grid:
        r = ranks.get(n.input[0])
        if r is not None and r != 4:
            bad.append((n.name, r))
    if bad:
        raise AssertionError(f"non-4D GridSample inputs (TRT GridSample is 4D-only): {bad}")
    print(f"[verify] all {n_grid} GridSample inputs are rank-4 (no 5D — TRT-native)")
    meta["gridsample_nodes"] = n_grid
    _verify_io(graph, meta)


def _verify_io(graph, meta: dict) -> None:
    in_names = [i.name for i in graph.input]
    out_names = [o.name for o in graph.output]
    print(f"[verify] inputs={in_names} outputs={out_names}")
    if in_names != ["images"] or out_names != ["logits", "boxes"]:
        raise AssertionError("unexpected graph I/O names")

    def batch_dim(value_info) -> str:
        d = value_info.type.tensor_type.shape.dim[0]
        return d.dim_param or str(d.dim_value)

    for vi in list(graph.input) + list(graph.output):
        bd = batch_dim(vi)
        print(f"[verify]   {vi.name} batch-dim = {bd!r}")
        if bd in ("", "1", "0"):
            raise AssertionError(f"{vi.name} batch dim is not symbolic ({bd!r}); dynamic batch lost")

    plugin_ops = {n.op_type for n in graph.node if n.domain not in ("", "ai.onnx")}
    print(f"[verify] non-standard-domain ops: {sorted(plugin_ops) or 'none'}")
    if plugin_ops:
        raise AssertionError(f"graph contains custom-domain ops (would need a TRT plugin): {plugin_ops}")
    print("[verify] graph OK: native ops only, symbolic batch, 2 raw outputs")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve()
    repo = here.parents[2]  # D-FINE-cpp/
    p = argparse.ArgumentParser(description="Export D-FINE to a raw two-output ONNX for D-FINE-cpp")
    p.add_argument("--model-name", default="m", choices=["n", "s", "m", "l", "x"])
    p.add_argument("--num-classes", type=int, default=80)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--max-batch", type=int, default=8)
    p.add_argument("--trace-batch", type=int, default=2,
                   help="batch size of the tracing dummy; must be >=2 to keep the batch axis dynamic")
    p.add_argument("--opset", type=int, default=16)
    p.add_argument("--checkpoint", required=True, help="path to a D-FINE detection .pt/.pth")
    p.add_argument("--class-names", default="",
                   help="display names for the sidecar: a file (one name per line) or a comma "
                        "list; must match --num-classes. Default: COCO-80 when num_classes==80")
    p.add_argument("--dfine-src",
                   default=os.environ.get("DFINE_SEG_SRC",
                                          "/home/dxdxxd/projects/custom-dfine/D-FINE-seg"),
                   help="root of the D-FINE-seg source (github.com/ArgoHA/D-FINE-seg) providing "
                        "build_model; or set $DFINE_SEG_SRC")
    p.add_argument("--output", default=str(repo / "trt-files" / "onnx" / "dfine_m.onnx"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-simplify", action="store_true")
    p.add_argument("--deform", default="explicit", choices=["explicit", "gridsample"],
                   help="explicit gather-bilinear (TRT-accurate, default) vs native GridSample (~10 AP loss on TRT)")
    return p.parse_args()


if __name__ == "__main__":
    export(parse_args())
