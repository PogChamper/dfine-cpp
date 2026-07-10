#!/usr/bin/env python3
"""Surgical FP16: whole net FP16 INCLUDING decoder modules; FP32 only for the
FDR-critical subset (sibling of convert_fp16.py; needs an opset >= 19 export).

FP32 island (from the E2 torch ablation that measured ~-0.001 AP for this split):
  1. FDR tail scopes: integral, lqe_layers, dec_bbox_head.*, pre_bbox_head,
     enc_bbox_head (reference-point producers included).
  2. Decoder functional glue: leaf ops directly under /model/decoder/ and
     /model/decoder/decoder/ (anchor math, inverse_sigmoid, pred_corners
     accumulation Adds, distance2bbox chain, ref-point sigmoids, dense-head topk).
     ``--slim`` drops this tier to FP16 too — measured lossless on COCO full-val
     for all five sizes, +2-3% b8 throughput — and is the release default.
  3. Deform coordinate chain: ancestors of every cross_attn gather INDEX input,
     walked through pointwise/shape ops, stopping at MatMul/Gemm/Softmax module
     compute — keeps unnormalize/floor/clip/flatten-index math FP32 (F-1: index
     arithmetic >2048 is inexact in fp16) while the data path (value gathers,
     bilinear/attention weight multiplies) goes FP16.
Everything else — backbone, encoder, self_attn, cross_attn projections, FFN,
gateway, norms, query_pos/score heads — becomes FP16.

Opset >= 19 is REQUIRED: opset-16 exports decompose LayerNorm into primitive ops,
and TensorRT miscompiles that decomposition in FP16 (mAP collapses to ~0.005;
ONNXRuntime stays healthy — a TRT-side bug, repro archived). Opset 19 exports a
native LayerNormalization node, which compiles correctly.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import onnx
from onnx import TensorProto, helper
from onnxconverter_common import float16

FLOAT_TYPES = (TensorProto.FLOAT, TensorProto.FLOAT16)
STOP_OPS = {"MatMul", "Gemm", "Softmax", "Conv"}
import os
_EXTRA = tuple(x for x in os.environ.get("SURGICAL_EXTRA_SCOPES", "").split(",") if x)
FDR_SCOPES = _EXTRA + (
    "/model/decoder/decoder/integral",
    "/model/decoder/decoder/lqe",
    "dec_bbox_head",
    "/model/decoder/pre_bbox_head",
    "/model/decoder/enc_bbox_head",
)



def _publish_pair(graph_tmp, graph_out, sidecar_text, sidecar_out, tag):
    """Publish a staged graph and its (optional) sidecar with the smallest
    possible inconsistency window: the sidecar is staged BEFORE either swap,
    then both land via two adjacent atomic renames (each rename atomic; the
    pair is not jointly transactional — the window is two syscalls).
    sidecar_text=None means this producer has no contract to carry through;
    a sidecar already sitting at sidecar_out would then describe the PREVIOUS
    graph, so it is removed in the same publish step."""
    graph_tmp, graph_out = Path(graph_tmp), Path(graph_out)
    sidecar_out = Path(sidecar_out)
    sc_tmp = None
    if sidecar_text is not None:
        sc_tmp = Path(str(sidecar_out) + ".tmp")
        sc_tmp.write_text(sidecar_text)
    os.replace(graph_tmp, graph_out)
    if sc_tmp is not None:
        os.replace(sc_tmp, sidecar_out)
    elif sidecar_out.exists():
        sidecar_out.unlink()
        print(f"[{tag}] removed stale sidecar {sidecar_out} (source has none)")

def is_glue_leaf(name: str) -> bool:
    """Leaf op directly under /model/decoder/ or /model/decoder/decoder/."""
    for prefix in ("/model/decoder/decoder/", "/model/decoder/"):
        if name.startswith(prefix):
            rest = name[len(prefix):]
            return "/" not in rest
    return False


def coord_slice(g, by_output) -> set[str]:
    """Deform coordinate/index chain: ancestors of every cross_attn gather INDEX
    input, stopping at module compute (MatMul/Gemm/Softmax/Conv). Must stay FP32
    (F-1: index integers >2048 are inexact in fp16)."""
    frontier = []
    for n in g.node:
        if "cross_attn" in n.name and n.op_type in ("Gather", "GatherElements", "GatherND"):
            if len(n.input) > 1:
                frontier.append(n.input[1])
    seen_t: set[str] = set()
    out: set[str] = set()
    while frontier:
        t = frontier.pop()
        if t in seen_t:
            continue
        seen_t.add(t)
        p = by_output.get(t)
        if p is None or "/model/decoder" not in p.name or p.name in out:
            continue
        out.add(p.name)
        if p.op_type in STOP_OPS:
            continue
        frontier.extend(p.input)
    return out


def build_blocklist(model: onnx.ModelProto, slim: bool = False) -> list[str]:
    g = model.graph
    by_output: dict[str, onnx.NodeProto] = {}
    for n in g.node:
        for o in n.output:
            by_output[o] = n

    block: set[str] = set()
    n_scope = n_glue = 0
    fp16_only = tuple(x for x in os.environ.get("SURGICAL_FP16_ONLY", "").split(",") if x)
    if fp16_only:
        # coarse mode: EVERYTHING under the decoder is FP32 except the named
        # contiguous module scopes (minimizes fp16/fp32 boundary count)
        for n in g.node:
            if ("/model/decoder" in n.name or "model.decoder" in n.name) and \
                    not any(s in n.name for s in fp16_only):
                block.add(n.name)
                n_scope += 1
        n_coord = 0
        if any("cross_attn" in s for s in fp16_only):
            # hybrid: cross_attn data path fp16, coordinate/index math re-blocked FP32
            cs = coord_slice(g, by_output)
            n_coord = len(cs - block)
            block |= cs
        print(f"[surgical] coarse mode: decoder FP32 except {fp16_only}"
              + (f" (+{n_coord} coord-slice nodes re-blocked FP32)" if n_coord else ""))
        return sorted(block)
    for n in g.node:
        if any(s in n.name for s in FDR_SCOPES):
            block.add(n.name)
            n_scope += 1
        elif not slim and is_glue_leaf(n.name):
            block.add(n.name)
            n_glue += 1

    # deform coordinate slice: ancestors of gather indices inside cross_attn
    cs = coord_slice(g, by_output)
    n_coord = len(cs - block)
    block |= cs

    print(f"[surgical] blocklist{' (slim: glue leaves stay FP16)' if slim else ''}: "
          f"{n_scope} FDR-scope + {n_glue} glue-leaf + "
          f"{n_coord} deform-coordinate nodes = {len(block)} total "
          f"(of {len(g.node)} graph nodes)")
    return sorted(block)


def retype_outputs_fp32(model: onnx.ModelProto) -> None:
    """Same contract as convert_fp16.py: graph outputs stay FP32."""
    g = model.graph
    prod = {}
    for n in g.node:
        for o in n.output:
            prod[o] = n
    for out in g.output:
        if out.type.tensor_type.elem_type != TensorProto.FLOAT16:
            continue
        p = prod.get(out.name)
        if p is not None and p.op_type == "Cast":
            for a in p.attribute:
                if a.name == "to":
                    a.i = TensorProto.FLOAT
        elif p is not None:
            inner = out.name + "_fp16out"
            for i, o in enumerate(p.output):
                if o == out.name:
                    p.output[i] = inner
            g.node.append(helper.make_node("Cast", [inner], [out.name],
                                           name=out.name + "_to_fp32",
                                           to=TensorProto.FLOAT))
        out.type.tensor_type.elem_type = TensorProto.FLOAT


# ops whose outputs are never float regardless of inputs
NONFLOAT_OUT = {"Shape", "ArgMax", "ArgMin", "Equal", "Less", "Greater",
                "LessOrEqual", "GreaterOrEqual", "And", "Or", "Not", "NonZero"}


def _attr_tensor_dtype(n: onnx.NodeProto):
    for a in n.attribute:
        if a.name == "value" and a.HasField("t"):
            return a.t.data_type
    return None


def harmonize_blockset(model: onnx.ModelProto, block: set[str]) -> int:
    """Topological true-type propagation + fix. The converter's value_info is
    STALE after node_block_list conversion (blocked nodes keep FP32 at runtime
    but annotations still say FP16), so types are derived from the graph itself:
    blocked node -> FLOAT, unblocked -> FLOAT16, Cast/Constant by attribute.
    Any node with float inputs that mismatch its target gets input casts (or a
    dtype-duplicated initializer)."""
    import numpy as np

    g = model.graph
    del g.value_info[:]  # all stale after this pass; checker doesn't need them
    vtype: dict[str, int] = {}
    for vi in list(g.input):
        vtype[vi.name] = vi.type.tensor_type.elem_type
    inits = {i.name: i for i in g.initializer}
    for i in g.initializer:
        vtype[i.name] = i.data_type

    fixed = 0
    dup_cache: dict[tuple[str, int], str] = {}
    pending: list[tuple[int, onnx.NodeProto]] = []
    for idx, n in enumerate(g.node):
        # resolve this node's float target
        if n.op_type == "Cast":
            to = next(a.i for a in n.attribute if a.name == "to")
            vtype[n.output[0]] = to
            continue
        if n.op_type in ("Constant", "ConstantOfShape"):
            dt = _attr_tensor_dtype(n)
            if dt is not None:
                vtype[n.output[0]] = dt
            continue
        # spec-pinned inputs that must stay float32 regardless of the data dtype
        SPEC_F32 = {"Resize": (1, 2)}  # roi, scales
        skip = SPEC_F32.get(n.op_type, ())
        float_ins = [(i, vtype.get(x)) for i, x in enumerate(n.input)
                     if i not in skip and vtype.get(x) in FLOAT_TYPES]
        target = TensorProto.FLOAT if n.name in block else TensorProto.FLOAT16
        for i, t in float_ins:
            if t == target:
                continue
            iname = n.input[i]
            if iname in inits:
                key = (iname, target)
                if key not in dup_cache:
                    arr = onnx.numpy_helper.to_array(inits[iname])
                    arr = arr.astype(np.float16 if target == TensorProto.FLOAT16
                                     else np.float32)
                    suffix = "__f16" if target == TensorProto.FLOAT16 else "__f32"
                    ni = onnx.numpy_helper.from_array(arr, iname + suffix)
                    g.initializer.append(ni)
                    vtype[iname + suffix] = target
                    dup_cache[key] = iname + suffix
                n.input[i] = dup_cache[key]
            else:
                cast_out = f"{iname}__harm{fixed}"
                cast = helper.make_node("Cast", [iname], [cast_out],
                                        name=f"harmonize_cast_{fixed}", to=target)
                vtype[cast_out] = target
                pending.append((idx, cast))
                n.input[i] = cast_out
            fixed += 1
        # propagate output types
        out_dt = None
        if n.op_type in NONFLOAT_OUT:
            out_dt = None  # bool/int outputs — leave untyped, never float-fixed
        elif float_ins:
            out_dt = target
        for oi, o in enumerate(n.output):
            if n.op_type == "TopK" and oi == 1:
                vtype[o] = TensorProto.INT64
            elif out_dt is not None:
                vtype[o] = out_dt
    for idx, cast in reversed(pending):
        g.node.insert(idx, cast)
    return fixed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--slim", action="store_true",
                   help="leave the decoder glue leaves FP16 too (FP32 island = FDR scopes "
                        "+ deform coordinate slice only) — measured lossless on COCO "
                        "full-val for all five sizes, +2-3%% b8; the release default")
    args = p.parse_args()
    env_slim = os.environ.get("SURGICAL_NO_GLUE", "").strip().lower()
    slim = args.slim or env_slim not in ("", "0", "false", "no", "off")

    model = onnx.load(args.onnx)
    opset = max((imp.version for imp in model.opset_import
                 if imp.domain in ("", "ai.onnx")), default=0)
    if opset < 19 and not os.environ.get("SURGICAL_FP16_ONLY"):
        raise SystemExit(
            f"[surgical] input opset is {opset}, need >= 19: opset-16 exports decompose "
            "LayerNorm and TensorRT miscompiles the decomposition in FP16 (mAP ~0.005). "
            "Re-export with export_dfine_onnx.py --opset 19.")
    block = build_blocklist(model, slim=slim)
    model16 = float16.convert_float_to_float16(
        model, node_block_list=block, keep_io_types=True, disable_shape_infer=False)
    retype_outputs_fp32(model16)
    n = 1
    total = 0
    while n:  # new casts can expose new mixed nodes; iterate to fixpoint
        n = harmonize_blockset(model16, set(block))
        total += n
        if total > 5000:
            raise RuntimeError("harmonize did not converge")
    print(f"[surgical] harmonized {total} mixed-type inputs")
    onnx.checker.check_model(model16)
    tmp = args.output + ".tmp"
    onnx.save(model16, tmp)
    src_sidecar = Path(args.onnx).with_suffix(".json")
    sidecar_text = None
    if src_sidecar.exists():
        meta = json.loads(src_sidecar.read_text())
        meta["precision"] = "fp16"
        meta["precision_mode"] = ("strongly_typed_onnx_fp16_surgical_slim" if slim
                                  else "strongly_typed_onnx_fp16_surgical_decoder")
        sidecar_text = json.dumps(meta, indent=2) + "\n"
    _publish_pair(tmp, args.output, sidecar_text,
                  Path(args.output).with_suffix(".json"), "surgical")
    print(f"[surgical] wrote {args.output}")


if __name__ == "__main__":
    main()
