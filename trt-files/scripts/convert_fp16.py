#!/usr/bin/env python3
"""Convert a raw D-FINE ONNX to mixed FP16 (backbone+encoder FP16, decoder FP32).

This is the *strongly-typed* path to FP16, and it exists because the weakly-typed
`config.set_flag(kFP16)` route degrades D-FINE by ~6.8 AP even with every compute
layer pinned FP32 — TRT inserts uncontrolled FP16 reformats on the FDR's
precision-critical data path (docs/HANDOFF M2.1). Here the precision is baked into
the ONNX tensor types instead: onnxconverter_common casts backbone+encoder tensors to
FP16 and inserts explicit Cast nodes at the decoder boundary, and the decoder
(block-listed by name) stays FP32. Build the result with `build_engine.py
--strongly-typed` (NO kFP16 flag), so TRT reproduces exactly these types.

    convert_fp16.py --onnx dfine_m.onnx --output dfine_m_fp16_st.onnx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper
from onnxconverter_common import float16


def keep_fp32_nodes(model: onnx.ModelProto, prefixes: tuple[str, ...]) -> list[str]:
    return [n.name for n in model.graph.node if n.name.startswith(prefixes)]


def harmonize_float_types(model: onnx.ModelProto, decoder_prefixes: tuple[str, ...]) -> int:
    """Strongly-typed TensorRT rejects an elementwise/matmul node whose float inputs mix
    Half and Float. onnxconverter_common occasionally leaves such a node behind (a
    size-dependent heuristic — a stray FP32 attention-scale constant in the FP16 encoder,
    or a missing FP16->FP32 boundary cast into a block-listed decoder node). Make every
    multi-input float node type-consistent: a node's target float type is FP32 if it is
    decoder-scoped (block-listed) else FP16. A mismatched *constant* is DUPLICATED as a
    target-typed copy for this consumer (the attention scale is shared across the FP16
    encoder and the FP32 decoder, so it can't be retyped in place); a mismatched
    *activation* gets a Cast inserted right before the node. Runs shape inference first so
    activation types are known. Returns the count fixed."""
    model.CopyFrom(onnx.shape_inference.infer_shapes(model))
    g = model.graph
    F = (TensorProto.FLOAT, TensorProto.FLOAT16)
    vtype = {vi.name: vi.type.tensor_type.elem_type
             for vi in list(g.value_info) + list(g.input) + list(g.output)}
    inits = {i.name: i for i in g.initializer}

    def elem_type(name):
        return inits[name].data_type if name in inits else vtype.get(name)

    shared = {"Mul", "Add", "Sub", "Div", "Pow", "Min", "Max", "MatMul", "Gemm", "Where"}
    fixed = 0
    dup = {}            # (init_name, target) -> duplicated init name
    pending_casts = []  # (node_index, cast_node) — inserted before the consumer (topo order)
    for ni, node in enumerate(g.node):
        if node.op_type not in shared:
            continue
        types = {elem_type(i) for i in node.input if elem_type(i) in F}
        if len(types) < 2:
            continue
        target = (TensorProto.FLOAT if node.name.startswith(decoder_prefixes)
                  else TensorProto.FLOAT16)
        npd = np.float16 if target == TensorProto.FLOAT16 else np.float32
        for idx, iname in enumerate(node.input):
            if elem_type(iname) not in F or elem_type(iname) == target:
                continue
            if iname in inits:  # duplicate the constant as a target-typed copy for this node
                key = (iname, target)
                if key not in dup:
                    newname = f"{iname}__{'f16' if target == TensorProto.FLOAT16 else 'f32'}"
                    g.initializer.append(numpy_helper.from_array(
                        numpy_helper.to_array(inits[iname]).astype(npd), newname))
                    inits[newname] = g.initializer[-1]
                    dup[key] = newname
                node.input[idx] = dup[key]
            else:               # activation: insert a boundary Cast
                cast_out = f"{iname}__harm{fixed}"
                pending_casts.append((ni, onnx.helper.make_node(
                    "Cast", [iname], [cast_out], to=target, name=f"harmonize_cast_{fixed}")))
                node.input[idx] = cast_out
                vtype[cast_out] = target
            fixed += 1
    for ni, cast in sorted(pending_casts, key=lambda x: -x[0]):
        g.node.insert(ni, cast)
    return fixed


def main(args: argparse.Namespace) -> None:
    onnx_path = Path(args.onnx).resolve()
    out_path = Path(args.output).resolve()
    model = onnx.load(str(onnx_path))

    prefixes = tuple(p for p in args.fp32_prefixes.split(",") if p)
    block = keep_fp32_nodes(model, prefixes)
    print(f"[fp16] keeping {len(block)} nodes FP32 (prefixes={list(prefixes)}); converting the rest to FP16")

    # keep_io_types keeps the input FP32 but still appends a trailing Cast-to-FP16 on
    # each graph output. The decoder (block-listed) already emits FP32, so retype the
    # outputs back to FP32 — profile.py's trt backend and CUDA-graph replay want FP32
    # outputs, and it just drops a redundant downcast.
    model16 = float16.convert_float_to_float16(
        model, node_block_list=block, keep_io_types=True, disable_shape_infer=False)

    from onnx import TensorProto, helper
    prod = {o: n for n in model16.graph.node for o in n.output}
    for out in model16.graph.output:
        if out.type.tensor_type.elem_type != TensorProto.FLOAT16:
            continue
        n = prod.get(out.name)
        if n is not None and n.op_type == "Cast":
            for a in n.attribute:
                if a.name == "to":
                    a.i = TensorProto.FLOAT  # FP16 downcast -> FP32 no-op (TRT elides it)
        else:
            inner = out.name + "_fp16out"
            for nn in model16.graph.node:
                nn.output[:] = [inner if o == out.name else o for o in nn.output]
            model16.graph.node.append(helper.make_node("Cast", [inner], [out.name], to=TensorProto.FLOAT))
        out.type.tensor_type.elem_type = TensorProto.FLOAT
    n_harmonized = harmonize_float_types(model16, prefixes)
    if n_harmonized:
        print(f"[fp16] harmonized {n_harmonized} mixed Half/Float node inputs "
              "(strongly-typed TRT would reject them)")
    onnx.checker.check_model(model16)
    onnx.save(model16, str(out_path))

    n_cast = sum(1 for n in model16.graph.node if n.op_type == "Cast")
    print(f"[fp16] wrote {out_path} ({n_cast} Cast nodes at precision boundaries)")

    side = onnx_path.with_suffix(".json")
    if side.exists():
        meta = json.loads(side.read_text())
        meta["precision"] = "fp16"
        meta["fp16_decoder_fp32"] = True
        meta["precision_mode"] = "strongly_typed_onnx_fp16"
        out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"[fp16] wrote sidecar {out_path.with_suffix('.json')}")


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="ONNX-level mixed FP16 (decoder kept FP32) for strong typing")
    p.add_argument("--onnx", default=str(repo / "trt-files" / "onnx" / "dfine_m.onnx"))
    p.add_argument("--output", default=str(repo / "trt-files" / "onnx" / "dfine_m_fp16_st.onnx"))
    p.add_argument("--fp32-prefixes", default="/model/decoder,model.decoder",
                   help="node-name prefixes to keep FP32 (default: the whole decoder)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
