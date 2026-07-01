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

import onnx
from onnxconverter_common import float16


def keep_fp32_nodes(model: onnx.ModelProto, prefixes: tuple[str, ...]) -> list[str]:
    return [n.name for n in model.graph.node if n.name.startswith(prefixes)]


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
