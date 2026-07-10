#!/usr/bin/env python3
"""Export a D-FINE detection model to a RAW two-output ONNX graph for D-FINE-cpp.

The exported graph takes a single ``images`` input and returns the decoder's raw
``logits`` (pre-sigmoid, [N, num_queries, num_classes]) and ``boxes`` (normalized
cxcywh, [N, num_queries, 4]). Sigmoid + top-k + box conversion are intentionally
left out of the graph and performed in C++.

The FDR/Integral/LQE box decode and the deformable attention stay inside the graph
regardless of this choice; ``model.deploy()`` folds the weighting vector and truncates
the decoder to its evaluation layers.

Optional accuracy/speed sliders (``--num-queries``, ``--eval-idx``, ``--cascade``)
reshape the decoder before deploy/tracing; the measured cost/gain of each (and of the
composed ``fast``/``max`` presets) is tabulated in docs/RESEARCH_MATRIX.md.

A JSON sidecar with the same stem describes the engine contract (input geometry,
normalization, per-variant constants) so the C++ runtime stays model-generic.

Model construction uses the D-FINE-seg package, whose detection subnetwork is
architecturally identical to authorial D-FINE;
the resulting graph is representative of either repo.
"""

from __future__ import annotations

import argparse
import hashlib
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


# --- Export-time accuracy/speed sliders ----------------------------------------------
# Three optional decoder reshapes, applied to the torch model AFTER the checkpoint is
# loaded and BEFORE model.deploy(), so the exported graph itself is smaller — nothing is
# masked at runtime. Measured trade-offs (COCO full-val, RTX 4070 Ti SUPER, batch 8):
# docs/RESEARCH_MATRIX.md. All three compose (the `fast`/`max` presets in the README).


def _parse_cascade(spec: str) -> tuple[int, int]:
    try:
        k, keep = (int(v) for v in spec.split(":"))
    except ValueError:
        raise SystemExit(f"--cascade wants K:KEEP (e.g. 1:150), got {spec!r}") from None
    return k, keep


def patch_cascade(model: nn.Module, k: int, keep: int) -> None:
    """Prune to the top-``keep`` queries after decoder layer ``k``.

    Queries are ranked by layer ``k``'s trained deep-supervision score head — a
    ranking head the standard deploy path folds away, which is why the head is
    deep-copied here, before ``model.deploy()`` truncates the aux heads. The decoder
    forward is replaced by a deploy-mode equivalent that TopK+Gathers every per-query
    tensor after layer ``k``, so layers ``k+1..eval_idx`` (self-attention is O(Q²))
    and the output decode run on ``keep`` queries instead of ``num_queries``.
    """
    import copy
    import types

    import torch.nn.functional as F

    chead = copy.deepcopy(model.decoder.dec_score_head[k]).eval().requires_grad_(False)

    def _cascade_forward(self, target, ref_points_unact, memory, spatial_shapes,
                         bbox_head, score_head, query_pos_head, pre_bbox_head,
                         integral, up, reg_scale, attn_mask=None, memory_mask=None,
                         return_queries=False):
        from src.d_fine.arch.utils import distance2bbox, inverse_sigmoid
        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)
        project = self.project
        ref_points_detach = F.sigmoid(ref_points_unact)
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)
            output = layer(output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed)
            if i == 0:
                pre_bboxes = F.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach))
                ref_points_initial = pre_bboxes.detach()
            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(ref_points_initial, integral(pred_corners, project),
                                           reg_scale, deploy=True)
            if i == self.eval_idx:
                scores = score_head[i](output)
                scores = self.lqe_layers[i](scores, pred_corners)
                return (inter_ref_bbox.unsqueeze(0), scores.unsqueeze(0),
                        pred_corners.unsqueeze(0), ref_points_initial.unsqueeze(0),
                        pre_bboxes, None, None)
            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()
            if i == k:
                rank = F.sigmoid(chead(output)).amax(-1)                # [B, Q]
                keep_idx = rank.topk(keep, dim=1).indices               # [B, keep]

                def _g(t):
                    return t.gather(1, keep_idx.unsqueeze(-1).expand(-1, -1, t.shape[-1]))

                output = _g(output)
                output_detach = _g(output_detach)
                pred_corners_undetach = _g(pred_corners_undetach)
                ref_points_detach = _g(ref_points_detach)
                ref_points_initial = _g(ref_points_initial)
        raise RuntimeError("cascade forward: eval_idx not reached")

    model.decoder.decoder.forward = types.MethodType(_cascade_forward, model.decoder.decoder)


def apply_sliders(model: nn.Module, args: argparse.Namespace) -> None:
    n_layers = len(model.decoder.decoder.layers)
    if args.eval_idx is not None:
        if not 0 <= args.eval_idx < n_layers:
            raise SystemExit(f"--eval-idx {args.eval_idx} out of range [0, {n_layers})")
        # deploy() truncates the layer stack to eval_idx+1, so this shrinks the graph
        model.decoder.eval_idx = args.eval_idx
        model.decoder.decoder.eval_idx = args.eval_idx
        print(f"[sliders] eval_idx -> {args.eval_idx} (deploy keeps {args.eval_idx + 1} decoder layers)")
    if args.num_queries is not None:
        if args.num_queries <= 0:
            raise SystemExit(f"--num-queries must be positive, got {args.num_queries}")
        model.decoder.num_queries = args.num_queries
        print(f"[sliders] num_queries -> {args.num_queries}")
    if args.cascade:
        k, keep = _parse_cascade(args.cascade)
        eval_idx = int(model.decoder.eval_idx)
        if eval_idx < 0:
            eval_idx += n_layers
        nq = int(model.decoder.num_queries)
        if not 0 <= k < eval_idx:
            raise SystemExit(f"--cascade layer K={k} must satisfy 0 <= K < eval_idx ({eval_idx}); "
                             "pruning at or after the scoring layer is a no-op")
        if not 0 < keep < nq:
            raise SystemExit(f"--cascade KEEP={keep} must be in (0, num_queries={nq})")
        patch_cascade(model, k, keep)
        print(f"[sliders] cascade: prune to top-{keep} queries after decoder layer {k}")


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
    cascade = _parse_cascade(args.cascade) if args.cascade else None
    # With a cascade the graph outputs KEEP queries, not the decoder's initial count —
    # the sidecar must describe the engine contract, so num_queries follows the output.
    num_queries_out = cascade[1] if cascade else int(dec.num_queries)
    return {
        "model": "d-fine",
        "variant": args.model_name,
        "task": "detect",
        "input_h": args.img_size,
        "input_w": args.img_size,
        "num_classes": args.num_classes,
        "num_queries": num_queries_out,
        **({"cascade": args.cascade,
            "cascade_initial_queries": int(dec.num_queries)} if cascade else {}),
        "reg_max": int(dec.reg_max),
        "reg_scale": round(_scalar(dec.reg_scale), 6),
        "num_decoder_layers": len(dec.decoder.layers),
        "eval_idx": eval_idx,
        "num_levels": int(dec.num_levels),
        "hidden_dim": int(dec.hidden_dim),
        "feat_strides": list(getattr(enc, "feat_strides", [])),
        "input_names": ["images"],
        "output_names": ["logits", "boxes"],
        "logits_shape": ["N", num_queries_out, args.num_classes],
        "boxes_shape": ["N", num_queries_out, 4],
        "box_format": "cxcywh_normalized",
        "score_activation": "sigmoid",
        "color_order": "RGB",
        "channel_layout": "NCHW",
        "normalize": "div255",
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "resize": args.resize,
        **({"letterbox_anchor": args.letterbox_anchor,
            "letterbox_pad": args.letterbox_pad,
            "letterbox_upscale": not args.no_letterbox_upscale}
           if args.resize == "letterbox" else {}),
        "nms": "none",
        "has_masks": False,
        "dynamic_batch": True,
        "max_batch": args.max_batch,
        "trace_batch": args.trace_batch,
        "opset": args.opset,
        # Always present so no downstream tool ever has to GUESS the compute
        # types: the FP16 converters overwrite both fields with their recipe.
        "precision": "fp32",
        "precision_mode": "fp32",
        "deform_core": args.deform,
        "trt_min_version": "8.5",
        **({"class_names": class_names} if class_names else {}),
    }


# --- Checkpoint loading ---------------------------------------------------------------
# Strict by default: every model parameter/buffer must be filled from the checkpoint
# with an exactly matching shape, or the export stops before tracing. The upstream
# fine-tuning loader (load_tuning_state) silently drops every missing/mismatched key —
# a wrong --model-name or a forgotten --num-classes then exports a checker-clean ONNX
# whose unfilled weights are random initialization.


def _select_state(raw: dict) -> tuple[dict, str]:
    """Pick the model weights out of a training checkpoint: the EMA weights first
    (what upstream evaluates and deploys), then the plain model, then the mapping
    itself (a bare state dict)."""
    ema = raw.get("ema")
    if isinstance(ema, dict) and isinstance(ema.get("module"), dict):
        return ema["module"], "ema.module"
    if isinstance(raw.get("model"), dict):
        return raw["model"], "model"
    return raw, "checkpoint root"


def _diff_state(model_sd: dict, state: dict) -> dict:
    """Compare a candidate state dict against the model's: which model tensors are
    missing, which have the wrong shape, and which checkpoint tensors are unused.
    Non-tensor entries (schedulers, counters) are ignored, not errors."""
    tensors = {k: v for k, v in state.items() if hasattr(v, "shape")}
    missing = [k for k in model_sd if k not in tensors]
    mismatched = [
        (k, tuple(tensors[k].shape), tuple(model_sd[k].shape))
        for k in model_sd
        if k in tensors and tuple(tensors[k].shape) != tuple(model_sd[k].shape)
    ]
    extra = [k for k in tensors if k not in model_sd]
    return {"missing": missing, "shape_mismatch": mismatched, "extra": extra}


def _mismatch_hints(diff: dict, model_sd_len: int) -> list[str]:
    """Actionable guesses about WHY the checkpoint does not fit."""
    hints = []
    for k, got, want in diff["shape_mismatch"]:
        if "score_head" in k and k.endswith(".weight") and got and want and got[0] != want[0]:
            hints.append(f"  hint: the checkpoint's score head has {got[0]} classes; "
                         f"export with --num-classes {got[0]} and matching --class-names")
            break
    bad = len(diff["missing"]) + len(diff["shape_mismatch"])
    if bad > model_sd_len // 4:
        hints.append("  hint: most tensors do not fit — the checkpoint likely belongs to a "
                     "different variant; pass the --model-name it was trained as")
    return hints


def load_checkpoint_state(model: nn.Module, ckpt: Path, allow_partial: bool) -> dict:
    """Load `ckpt` into `model` and return a load report for the sidecar.

    Strict mode (default): any missing or shape-mismatched model tensor aborts the
    export with the offending keys and a hint. --allow-partial-checkpoint downgrades
    that to a printed report (research escape hatch); the sidecar then records the
    partial load so the artifact cannot silently pass as a full export. Extra
    checkpoint tensors (e.g. a seg head on a detection export) are reported but
    tolerated — they cannot corrupt the loaded model.
    """
    try:
        raw = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001  (older checkpoints carry pickled objects)
        raw = torch.load(str(ckpt), map_location="cpu")
    if not isinstance(raw, dict):
        raise SystemExit(f"checkpoint {ckpt} does not contain a state dict "
                         f"(got {type(raw).__name__})")
    state, selected = _select_state(raw)
    model_sd = model.state_dict()
    diff = _diff_state(model_sd, state)

    bad = diff["missing"] or diff["shape_mismatch"]
    if bad:
        lines = [f"[export] checkpoint does not match the model: {len(diff['missing'])} missing, "
                 f"{len(diff['shape_mismatch'])} shape-mismatched of {len(model_sd)} model "
                 f"tensors (weights read from: {selected})"]
        lines += [f"  missing: {k}" for k in diff["missing"][:8]]
        lines += [f"  shape: {k} checkpoint{got} vs model{want}"
                  for k, got, want in diff["shape_mismatch"][:8]]
        hidden = len(diff["missing"]) + len(diff["shape_mismatch"]) - 16
        if hidden > 0:
            lines.append(f"  ... and {hidden} more")
        lines += _mismatch_hints(diff, len(model_sd))
        if not allow_partial:
            lines.append("  (--allow-partial-checkpoint loads what fits — research only; "
                         "the unfilled weights stay randomly initialized)")
            raise SystemExit("\n".join(lines))
        print("\n".join(lines))
        print("[export] --allow-partial-checkpoint: continuing with a PARTIAL load")

    loadable = {k: v for k, v in state.items()
                if k in model_sd and hasattr(v, "shape")
                and tuple(v.shape) == tuple(model_sd[k].shape)}
    model.load_state_dict(loadable, strict=False)
    if diff["extra"]:
        print(f"[export] note: {len(diff['extra'])} checkpoint tensors unused by this model "
              f"(e.g. {diff['extra'][0]})")
    return {
        "mode": "partial" if bad else "strict",
        "selected_state": selected,
        "loaded": len(loadable),
        "missing": len(diff["missing"]),
        "shape_mismatch": len(diff["shape_mismatch"]),
        "sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
    }


def build_detection_model(args: argparse.Namespace,
                          device: torch.device) -> tuple[nn.Module, dict]:
    from src.d_fine.dfine import build_model  # noqa: E402  (path set at runtime)

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
    load_report = load_checkpoint_state(model, ckpt, args.allow_partial_checkpoint)
    print(f"[export] checkpoint: {load_report['loaded']}/{len(model.state_dict())} tensors "
          f"loaded ({load_report['mode']}, from {load_report['selected_state']})")
    apply_sliders(model, args)
    return model.to(device), load_report


def export(args: argparse.Namespace) -> None:
    # Guarded here (not only in the parser) so programmatic callers cannot slip
    # through: a batch-1 trace bakes the anchor/GatherElements extent to 1, and
    # the resulting engine formally accepts dynamic N but only works at batch 1.
    if args.trace_batch < 2:
        raise SystemExit(f"--trace-batch must be >= 2 (got {args.trace_batch}): a batch-1 "
                         "trace bakes an internal decoder extent and breaks dynamic batch")
    # Fail before loading a checkpoint or overwriting an output. The execution
    # postcondition is part of a valid export, not a best-effort extra.
    _require_dynamic_batch_runtime()
    _add_repo_to_path(Path(args.dfine_src))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[export] device={device} variant={args.model_name} classes={args.num_classes}")

    model, load_report = build_detection_model(args, device)
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
    # Provenance: enough to answer "which weights, loaded how" for any artifact.
    meta["checkpoint_sha256"] = load_report["sha256"]
    meta["checkpoint_load"] = load_report["mode"]
    meta["checkpoint_loaded_tensors"] = load_report["loaded"]
    if load_report["mode"] == "partial":
        meta["checkpoint_missing_count"] = load_report["missing"]
        meta["checkpoint_shape_mismatch_count"] = load_report["shape_mismatch"]
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
        _verify_dynamic_batch_runs(onnx_path, meta)
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
    _verify_dynamic_batch_runs(onnx_path, meta)


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

    std = meta.get("std", [1.0, 1.0, 1.0])
    if any(not (isinstance(s, (int, float)) and s > 0) for s in std):
        raise AssertionError(f"sidecar std {std} must be positive "
                             "(a zero collapses the runtime normalization to inf)")
    print("[verify] graph OK: native ops only, symbolic batch, 2 raw outputs")


def _require_dynamic_batch_runtime():
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit(
            "dynamic-batch verification requires numpy and onnxruntime; install "
            "onnxruntime (or onnxruntime-gpu) before exporting"
        ) from exc
    return np, ort


def _verify_dynamic_batch_runs(onnx_path: Path, meta: dict) -> None:
    """The decisive dynamic-batch check: actually run the graph at N=1 and N=2.

    A batch-1 trace bakes an internal decoder extent while every STATIC check
    still passes (the graph I/O dims stay symbolic; the bake hides in folded
    constants) — only execution exposes it: the query-selection GatherElements
    rejects the second batch. The concrete output shapes double as the
    sidecar-consistency check (num_queries/num_classes really match the graph).
    """
    np, ort = _require_dynamic_batch_runtime()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    q, c = meta["num_queries"], meta["num_classes"]
    for n in (1, 2):
        x = rng.random((n, 3, meta["input_h"], meta["input_w"]), dtype=np.float32)
        try:
            logits, boxes = sess.run(None, {"images": x})
        except Exception as exc:  # noqa: BLE001  (ORT raises its own hierarchy)
            raise AssertionError(
                f"graph does not run at batch {n}: {exc}\nwith formally dynamic I/O this is "
                "the baked-extent signature — re-export with --trace-batch >= 2") from exc
        if tuple(logits.shape) != (n, q, c) or tuple(boxes.shape) != (n, q, 4):
            raise AssertionError(
                f"batch-{n} outputs logits{tuple(logits.shape)}/boxes{tuple(boxes.shape)} do not "
                f"match the sidecar contract [N,{q},{c}]/[N,{q},4] — wrong --num-classes/"
                "--num-queries, or the sidecar drifted from the graph")
    print(f"[verify] dynamic batch OK: N=1 and N=2 run; outputs match [N,{q},{c}] / [N,{q},4]")


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
    p.add_argument("--opset", type=int, default=19,
                   help="ONNX opset (default 19, the production base: native LayerNormalization, "
                        "required by the surgical FP16 converter; 16 = the legacy research base)")
    p.add_argument("--checkpoint", required=True, help="path to a D-FINE detection .pt/.pth")
    p.add_argument("--allow-partial-checkpoint", action="store_true",
                   help="research only: load whatever fits instead of aborting on a "
                        "missing/mismatched model tensor; the sidecar records the partial load")
    p.add_argument("--class-names", default="",
                   help="display names for the sidecar: a file (one name per line) or a comma "
                        "list; must match --num-classes. Default: COCO-80 when num_classes==80")
    p.add_argument("--resize", choices=["stretch", "letterbox"], default="stretch",
                   help="preprocessing geometry declared in the sidecar; the C++ runtime follows "
                        "it. D-FINE is trained with stretch (letterbox costs ~2 AP on the "
                        "published weights — see letterbox_eval.py)")
    p.add_argument("--letterbox-anchor", choices=["center", "topleft"], default="center")
    p.add_argument("--letterbox-pad", type=int, default=114)
    p.add_argument("--no-letterbox-upscale", action="store_true",
                   help="paste 1:1 when the frame already fits (production smart_resize)")
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
    sliders = p.add_argument_group(
        "accuracy/speed sliders",
        "optional decoder reshapes baked into the export; measured cost/gain tables in "
        "docs/RESEARCH_MATRIX.md (m/COCO full-val/b8: --num-queries 200 = -0.13 AP, "
        "--cascade 1:150 = -0.18 AP +8%, --eval-idx 2 = -0.57 AP; composed presets "
        "reach +21..46% throughput)")
    sliders.add_argument("--num-queries", type=int, default=None,
                         help="initial decoder queries (default: the checkpoint's 300); "
                              "200 halves the decode cost at -0.13 AP")
    sliders.add_argument("--eval-idx", type=int, default=None,
                         help="decoder layer that produces the output; deploy() drops "
                              "the layers after it (m: 2 keeps 3 of 4 layers, -0.57 AP)")
    sliders.add_argument("--cascade", default=None, metavar="K:KEEP",
                         help="after decoder layer K, keep only the top-KEEP queries "
                              "ranked by layer K's trained score head (1:150 = -0.18 AP, "
                              "+8%% b8 on m)")
    return p.parse_args()


if __name__ == "__main__":
    export(parse_args())
