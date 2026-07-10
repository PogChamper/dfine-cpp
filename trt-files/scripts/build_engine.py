#!/usr/bin/env python3
"""Build a TensorRT engine from a raw D-FINE ONNX with a dynamic-batch profile.

Mirrors what the C++ ``dfine_build`` app will do (native NvOnnxParser, one
optimization profile over the ``images`` batch axis), so the engine contract is
established here and reused by both Python verification and the C++ runtime.

Precision modes:
  (default)             FP32 (add ``--no-tf32`` for an FP32-faithful parity build).
  ``--fp16``            weakly-typed whole-graph FP16 — the *anti-example*: it lets
                        TRT run D-FINE's decoder in FP16, which the FDR integral
                        amplifies into ~10 AP loss (the same failure class as the
                        grid_sample trap, docs/impl/M0_STATUS.md). Timing only.
  ``--fp16-decoder-fp32`` production FP16: backbone+encoder in FP16, but every
                        float-compute decoder layer pinned to FP32 (class/bbox
                        heads, FDR Integral/LQE, distance2bbox, deformable core).
                        Decoder layers are found by ONNX-derived name prefix — all
                        D-FINE compute is cleanly scoped under /model/{backbone,
                        encoder,decoder} (verified: OTHER-region nodes are shape/
                        constant only). Pair with ``--no-tf32`` so the pinned FP32
                        decoder is also TF32-faithful. Outputs stay FP32, so the
                        C++ runtime needs no change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import tensorrt as trt

# Layer types that never carry float activations to pin — shape/constant plumbing.
# We still may pin a CONSTANT's *output type* (to keep a decoder weight FP32) but
# never its compute precision, which is meaningless for a constant.
_NO_COMPUTE_PRECISION = (trt.LayerType.CONSTANT, trt.LayerType.SHAPE,
                         trt.LayerType.ASSERTION, trt.LayerType.IDENTITY)



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

def pin_decoder_fp32(network: "trt.INetworkDefinition", prefixes: tuple[str, ...],
                     verbose: bool) -> int:
    """Force every float-compute decoder layer to FP32 while the rest of the graph is
    free to run FP16. Returns the number of layers pinned.

    The decoder is D-FINE's FP-sensitive region: the FDR Integral accumulates tiny
    upstream errors into box shifts (docs/impl/M0_STATUS.md), so letting TRT pick FP16
    there costs ~10 AP. We select decoder layers by name prefix; only layers with a
    float output are pinned, so shape/int plumbing under the same scope is left alone.
    Must be paired with OBEY/PREFER_PRECISION_CONSTRAINTS for TRT to honour it.
    """
    pinned = 0
    skipped = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if not layer.name.startswith(prefixes):
            continue
        float_outs = [j for j in range(layer.num_outputs)
                      if layer.get_output(j).dtype in (trt.DataType.FLOAT, trt.DataType.HALF)]
        if not float_outs:
            skipped += 1  # shape / int-only layer under the decoder scope
            continue
        if layer.type not in _NO_COMPUTE_PRECISION:
            layer.precision = trt.DataType.FLOAT
        for j in float_outs:
            layer.set_output_type(j, trt.DataType.FLOAT)  # keeps FP32 weights/activations
        pinned += 1
    print(f"[build] pinned {pinned} decoder layers to FP32 "
          f"(prefixes={list(prefixes)}, skipped {skipped} shape/int layers)")
    if verbose:
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.name.startswith(prefixes) and layer.precision_is_set:
                print(f"[build]   FP32-pinned: {layer.name} ({layer.type})")
    return pinned


def build(args: argparse.Namespace) -> None:
    onnx_path = Path(args.onnx).resolve()
    if not onnx_path.exists():
        raise FileNotFoundError(onnx_path)
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if sum(bool(x) for x in (args.fp16, args.fp16_decoder_fp32, args.bf16_decoder_fp32, args.int8)) > 1:
        raise SystemExit("--fp16, --fp16-decoder-fp32, --bf16-decoder-fp32 and --int8 are mutually exclusive")
    if (args.fp16_decoder_fp32 or args.bf16_decoder_fp32) and args.strongly_typed:
        # Per-layer setPrecision is rejected on a strongly-typed network (types come
        # from the ONNX). The mixed build is weakly-typed by construction.
        raise SystemExit("mixed-precision pinning needs a weakly-typed network; drop --strongly-typed")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 0
    if args.strongly_typed:
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        print("[build] strongly-typed network (precision pinned by ONNX types)")
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)

    print(f"[build] TensorRT {trt.__version__}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"[build][parser] {parser.get_error(i)}")
            raise RuntimeError("ONNX parse failed")
    print(f"[build] parsed {onnx_path.name}: "
          f"{network.num_inputs} inputs, {network.num_outputs} outputs, {network.num_layers} layers")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gb << 30)

    if args.max_aux_streams is not None:
        # TRT uses auxiliary streams for intra-inference parallelism by default (2 here).
        # A CUDA graph captured with cudaStreamCaptureModeThreadLocal only records the main
        # stream, silently missing aux-stream kernels -> an incomplete/incorrect graph. So a
        # graph-capturable engine must be built single-stream (--max-aux-streams 0); the
        # C++ detector gates capture on num_aux_streams()==0 exactly for this reason.
        config.max_aux_streams = args.max_aux_streams
        print(f"[build] max_aux_streams = {args.max_aux_streams}"
              + ("  (single-stream, CUDA-graph capturable)" if args.max_aux_streams == 0 else ""))

    if args.no_tf32:
        # TF32 is on by default and deviates ~1% from true FP32 PyTorch (the FDR
        # integral amplifies it into box error). Disable it for an FP32-faithful
        # parity reference; leave it on for production speed.
        config.clear_flag(trt.BuilderFlag.TF32)
        print("[build] TF32 disabled (FP32-faithful build)")

    if args.tactic:
        names = {"cublas": trt.TacticSource.CUBLAS, "cublaslt": trt.TacticSource.CUBLAS_LT,
                 "cudnn": trt.TacticSource.CUDNN, "edge": trt.TacticSource.EDGE_MASK_CONVOLUTIONS,
                 "jit": trt.TacticSource.JIT_CONVOLUTIONS}
        mask = 0
        for s in args.tactic.split(","):
            mask |= 1 << int(names[s.strip()])
        config.set_tactic_sources(mask)
        print(f"[build] tactic sources = {args.tactic}")

    if args.prefer_precision:
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        print("[build] PREFER_PRECISION_CONSTRAINTS set")
    if args.opt_level is not None:
        config.builder_optimization_level = args.opt_level
        print(f"[build] builder_optimization_level = {args.opt_level}")

    profile = builder.create_optimization_profile()
    inp = network.get_input(0)
    _, c, h, w = inp.shape
    profile.set_shape(inp.name,
                      (args.min_batch, c, h, w),
                      (args.opt_batch, c, h, w),
                      (args.max_batch, c, h, w))
    config.add_optimization_profile(profile)
    print(f"[build] profile {inp.name}: min={args.min_batch} opt={args.opt_batch} max={args.max_batch} (CHW={c}x{h}x{w})")

    if (args.fp16 or args.fp16_decoder_fp32) and not builder.platform_has_fast_fp16:
        print("[build] WARNING: platform reports no fast FP16")

    if args.int8:
        # Explicit QDQ: the ONNX (from convert_int8.py) already carries the Q/DQ nodes
        # and their calibrated scales, so no IInt8Calibrator is set (that is the
        # deprecated implicit path). TRT honours the QDQ placement; the decoder has no
        # Q/DQ, so it runs FP32. No kFP16 here — that would let TRT drop the decoder to
        # FP16.
        if not builder.platform_has_fast_int8:
            print("[build] WARNING: platform reports no fast INT8")
        config.set_flag(trt.BuilderFlag.INT8)
        print("[build] INT8 enabled (explicit QDQ from ONNX; decoder stays FP32)")

    if args.fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[build] FP16 enabled (weakly-typed, whole-graph — timing only, not production)")
    elif args.fp16_decoder_fp32 or args.bf16_decoder_fp32:
        # BF16 keeps FP32's exponent range (no overflow) with less mantissa precision;
        # FP16 is more precise but overflows on some activations. For D-FINE the encoder
        # attention overflows in FP16 on some images, so BF16 is the safer low-precision.
        low = trt.BuilderFlag.BF16 if args.bf16_decoder_fp32 else trt.BuilderFlag.FP16
        low_name = "BF16" if args.bf16_decoder_fp32 else "FP16"
        config.set_flag(low)
        # Pin the FP-sensitive layers to FP32, then tell TRT to honour it. OBEY is a hard
        # guarantee (build fails if unsatisfiable); PREFER is a soft hint TRT may override —
        # default OBEY because TRT will otherwise sneak low precision into pinned layers.
        constraint = (trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS if args.constraints == "obey"
                      else trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        config.set_flag(constraint)
        prefixes = tuple(p for p in args.decoder_prefixes.split(",") if p)
        n = pin_decoder_fp32(network, prefixes, args.verbose)
        if n == 0:
            raise RuntimeError(f"no layers matched prefixes {list(prefixes)} — "
                               "check the ONNX layer naming (see docs/HANDOFF M2.1)")
        print(f"[build] {low_name} mixed: unpinned layers {low_name}, {n} pinned FP32 "
              f"({args.constraints.upper()}_PRECISION_CONSTRAINTS)")

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("build_serialized_network returned None")
    # Atomic: an interrupted/OOM-killed build must not leave a truncated file at
    # the final path — the CLI cache treats existence as validity, so a partial
    # engine would poison the entry until manually deleted. The engine is staged
    # here and published only after its sidecar is staged too, so the pair lands
    # in two adjacent renames.
    tmp_path = Path(str(out_path) + ".tmp")
    tmp_path.write_bytes(plan)

    # Engine sidecar: the ONNX contract passes through untouched; the builder only
    # appends facts IT owns. Precision is decided by whoever set the compute types:
    # a weakly-typed flag mode here, or the converter's ONNX types (strongly typed) —
    # the builder must never overwrite the converter's recipe with a flag guess
    # (v0.3.0 stamped every strongly-typed FP16 engine "fp32" this way).
    sidecar = onnx_path.with_suffix(".json")
    meta = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    if not sidecar.exists():
        # No contract sidecar: record what the parsed network itself asserts, so
        # the runtime's sidecar-vs-engine cross-check sees the engine's real
        # dims/classes/queries instead of absent fields (class names and
        # normalization stay unknown — the runtime warns and uses its defaults).
        inp_shape = network.get_input(0).shape
        out_shapes = [tuple(network.get_output(i).shape) for i in range(network.num_outputs)]
        logits_shape = next((s for s in out_shapes if len(s) == 3 and s[-1] != 4), None)
        if len(inp_shape) == 4 and inp_shape[2] > 0 and inp_shape[3] > 0:
            meta["input_h"], meta["input_w"] = int(inp_shape[2]), int(inp_shape[3])
        if logits_shape and logits_shape[1] > 0 and logits_shape[2] > 0:
            meta["num_queries"], meta["num_classes"] = int(logits_shape[1]), int(logits_shape[2])
        print(f"[build] NOTE: no ONNX sidecar ({sidecar.name}) — engine sidecar carries the "
              "graph contract + build facts only (no class names/normalization)")
    if args.int8:
        meta["precision"], meta["precision_mode"] = "int8", "weakly_typed_int8_qdq"
        meta["fp16_decoder_fp32"] = False
    elif args.bf16_decoder_fp32:
        meta["precision"], meta["precision_mode"] = "bf16", "weakly_typed_bf16_decoder_fp32"
        meta["fp16_decoder_fp32"] = True
    elif args.fp16_decoder_fp32:
        meta["precision"], meta["precision_mode"] = "fp16", "weakly_typed_fp16_decoder_fp32"
        meta["fp16_decoder_fp32"] = True
    elif args.fp16:
        meta["precision"], meta["precision_mode"] = "fp16", "weakly_typed_fp16"
        meta["fp16_decoder_fp32"] = False
    else:
        # No flag changed compute types: the ONNX decides. Normalize legacy
        # sidecars (pre-v0.3.1 exports carry no precision_mode) without inventing
        # a recipe; a strongly-typed graph with no sidecar is honestly unknown,
        # not "fp32". fp16_decoder_fp32 likewise passes through untouched — the
        # legacy converter's decoder really does run FP32 (convert_fp16.py sets
        # it), and the builder has no better knowledge here.
        if args.strongly_typed:
            meta.setdefault("precision", "unknown" if not sidecar.exists() else "fp32")
            meta.setdefault("precision_mode", "strongly_typed_unknown"
                            if meta["precision"] != "fp32" else "fp32")
        else:
            meta.setdefault("precision", "fp32")
            meta.setdefault("precision_mode",
                            "fp32" if meta["precision"] == "fp32" else "strongly_typed_unknown")
    meta.update({
        "network_typing": "strong" if args.strongly_typed else "weak",
        "cuda_graph_compat": bool(args.cuda_graph),
        "trt_version": trt.__version__,
        "opt_batch": args.opt_batch,
        "min_batch": args.min_batch,
        "max_batch": args.max_batch,
        "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
    })
    # The engine sidecar must never clobber the ONNX's own sidecar: with a
    # same-stem output next to the ONNX (the README quickstart does exactly
    # this), with_suffix(".json") IS the ONNX sidecar. The runtime looks for
    # "<engine-path>.json" first, so the appended name stays resolvable.
    engine_sidecar = out_path.with_suffix(".json")
    if sidecar.exists() and engine_sidecar.resolve() == sidecar.resolve():
        engine_sidecar = Path(str(out_path) + ".json")
        print(f"[build] NOTE: engine and ONNX share a stem — engine sidecar goes to "
              f"{engine_sidecar.name} (the ONNX's own sidecar stays untouched)")
    _publish_pair(tmp_path, out_path, json.dumps(meta, indent=2) + "\n",
                  engine_sidecar, "build")
    print(f"[build] wrote {out_path} ({plan.nbytes / 1e6:.1f} MB)")
    print(f"[build] wrote engine sidecar {engine_sidecar} "
          f"(precision={meta['precision']}, mode={meta['precision_mode']})")


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Build a TensorRT engine from a raw D-FINE ONNX")
    p.add_argument("--onnx", default=str(repo / "trt-files" / "onnx" / "dfine_m.onnx"))
    p.add_argument("--output", default=str(repo / "trt-files" / "engines" / "dfine_m_fp32.engine"))
    p.add_argument("--min-batch", type=int, default=1)
    p.add_argument("--opt-batch", type=int, default=1)
    p.add_argument("--max-batch", type=int, default=8)
    p.add_argument("--workspace-gb", type=int, default=4)
    p.add_argument("--fp16", action="store_true",
                   help="naive whole-graph FP16 (anti-example: corrupts the decoder, timing only)")
    p.add_argument("--fp16-decoder-fp32", action="store_true",
                   help="mixed FP16: unpinned layers FP16, --decoder-prefixes layers pinned FP32")
    p.add_argument("--bf16-decoder-fp32", action="store_true",
                   help="mixed BF16 (no overflow, less mantissa): unpinned BF16, pinned FP32")
    p.add_argument("--int8", action="store_true",
                   help="INT8 from an explicit-QDQ ONNX (convert_int8.py); decoder stays FP32")
    p.add_argument("--decoder-prefixes", default="/model/decoder,model.decoder",
                   help="comma-separated layer-name prefixes identifying the decoder to pin FP32")
    p.add_argument("--constraints", choices=["obey", "prefer"], default="obey",
                   help="how strictly TRT honours the FP32 pins (--fp16-decoder-fp32)")
    p.add_argument("--no-tf32", action="store_true", help="disable TF32 for an FP32-faithful build")
    p.add_argument("--prefer-precision", action="store_true", help="PREFER_PRECISION_CONSTRAINTS")
    p.add_argument("--opt-level", type=int, default=None, help="builder_optimization_level 0-5")
    p.add_argument("--strongly-typed", action="store_true", help="strongly-typed network (pin FP32 by ONNX types)")
    p.add_argument("--tactic", default=None, help="restrict tactic sources, e.g. 'cublas' or 'cublas,edge,jit'")
    p.add_argument("--cuda-graph", action="store_true", help="label sidecar cuda_graph_compat=true (advisory)")
    p.add_argument("--max-aux-streams", type=int, default=None,
                   help="cap TRT auxiliary streams; 0 = single-stream (required for CUDA-graph capture)")
    p.add_argument("--verbose", action="store_true", help="list every FP32-pinned decoder layer")
    return p.parse_args()


if __name__ == "__main__":
    build(parse_args())
