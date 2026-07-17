#!/usr/bin/env python3
"""Build a TensorRT engine from a D-FINE ONNX graph.

The validated FP16 path is a surgically typed ONNX graph built with
``--strongly-typed``; see ``docs/CONVERSION.md``. The weakly typed ``--fp16`` and
decoder-pinning modes remain research controls. They measured about 6.8 AP below
the PyTorch reference and do not reproduce the typed-graph result. The separate
native-GridSample path measured about 10.5 AP below the reference.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import secrets
from pathlib import Path

import tensorrt as trt


def _sm_arch() -> str:
    """Compute capability of the build GPU ('89' for Ada). Engines are
    arch-specific, and nothing but the CLI cache filename recorded which arch a
    dev-tree engine was built for — the sidecar must."""
    try:
        cudart = ctypes.CDLL("libcudart.so.12")
        device = ctypes.c_int()
        major = ctypes.c_int()
        minor = ctypes.c_int()
        if cudart.cudaGetDevice(ctypes.byref(device)) != 0:
            return "unknown"
        if cudart.cudaDeviceGetAttribute(ctypes.byref(major), 75, device.value) != 0:
            return "unknown"
        if cudart.cudaDeviceGetAttribute(ctypes.byref(minor), 76, device.value) != 0:
            return "unknown"
        return f"{major.value}{minor.value}"
    except Exception:
        return "unknown"


# Layer types that never carry float activations to pin — shape/constant plumbing.
# We still may pin a CONSTANT's *output type* (to keep a decoder weight FP32) but
# never its compute precision, which is meaningless for a constant.
_NO_COMPUTE_PRECISION = (
    trt.LayerType.CONSTANT,
    trt.LayerType.SHAPE,
    trt.LayerType.ASSERTION,
    trt.LayerType.IDENTITY,
)


def _adjacent_temp(target: str | Path, suffix: str = ".tmp") -> Path:
    """Create a unique staging file on the target filesystem."""
    target = Path(target)
    try:
        existing_mode = target.stat().st_mode & 0o777
    except FileNotFoundError:
        existing_mode = None
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    for _ in range(128):
        path = target.parent / f".{target.name}.{secrets.token_hex(8)}{suffix}"
        try:
            mode = (existing_mode | 0o600) if existing_mode is not None else 0o666
            fd = os.open(path, flags, mode)
        except FileExistsError:
            continue
        try:
            if existing_mode is not None:
                os.fchmod(fd, existing_mode | 0o600)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        finally:
            os.close(fd)
        return path
    raise FileExistsError(f"cannot allocate adjacent staging file for {target}")


def _link_backup(path: Path) -> Path | None:
    """Preserve an existing output without copying a potentially large artifact."""
    try:
        path.lstat()
    except FileNotFoundError:
        return None
    backup = _adjacent_temp(path, ".previous")
    backup.unlink()
    try:
        if path.is_symlink():
            backup.symlink_to(os.readlink(path))
        else:
            os.link(path, backup)
    except BaseException:
        backup.unlink(missing_ok=True)
        raise
    return backup


def _restore_output(path: Path, backup: Path | None) -> None:
    if backup is None:
        path.unlink(missing_ok=True)
    else:
        os.replace(backup, path)


def _cleanup_publish_files(paths: tuple[Path | None, ...], tag: str) -> None:
    for path in paths:
        if path is None:
            continue
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            print(f"[{tag}] warning: cannot remove staging file {path}: {exc}")


def _publish_pair(
    graph_tmp,
    graph_out,
    sidecar_text,
    sidecar_out,
    tag,
    stale_sidecar: Path | None = None,
):
    """Publish a staged artifact pair with rollback and writer serialization.

    Cooperative producers are serialized, and ordinary publication failures
    restore the complete previous pair. The filesystem updates are individually
    atomic, not crash-transactional; interruption between them can leave a mixed
    artifact.
    """
    graph_tmp, graph_out = Path(graph_tmp), Path(graph_out)
    sidecar_out = Path(sidecar_out)
    stale_sidecar = Path(stale_sidecar) if stale_sidecar is not None else None
    outputs = (graph_out, sidecar_out) + ((stale_sidecar,) if stale_sidecar else ())
    parent = graph_out.parent.resolve()
    if graph_tmp.parent.resolve() != parent or any(
        path.parent.resolve() != parent for path in outputs
    ):
        raise ValueError("published artifacts and staging files must share one directory")
    if len({os.path.abspath(path) for path in (graph_tmp, *outputs)}) != 1 + len(outputs):
        raise ValueError("published artifact paths must be distinct")

    sidecar_tmp = None
    if sidecar_text is not None:
        sidecar_tmp = _adjacent_temp(sidecar_out)
        try:
            sidecar_tmp.write_text(sidecar_text)
        except BaseException:
            _cleanup_publish_files((graph_tmp, sidecar_tmp), tag)
            raise

    lock_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        lock_fd = os.open(parent, lock_flags)
    except BaseException:
        _cleanup_publish_files((graph_tmp, sidecar_tmp), tag)
        raise
    backups: dict[Path, Path | None] = {}
    changed: list[Path] = []
    preserve_backups = False
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        for staged, target in ((graph_tmp, graph_out), (sidecar_tmp, sidecar_out)):
            if staged is None:
                continue
            try:
                output_mode = target.stat().st_mode & 0o777
            except FileNotFoundError:
                continue
            staged.chmod(output_mode)
        for path in outputs:
            backups[path] = _link_backup(path)
        try:
            changed.append(graph_out)
            os.replace(graph_tmp, graph_out)
            changed.append(sidecar_out)
            if sidecar_tmp is None:
                sidecar_out.unlink(missing_ok=True)
            else:
                os.replace(sidecar_tmp, sidecar_out)
            if stale_sidecar is not None:
                changed.append(stale_sidecar)
                stale_sidecar.unlink(missing_ok=True)
        except BaseException as publish_error:
            rollback_errors = []
            for path in reversed(changed):
                try:
                    _restore_output(path, backups.get(path))
                except OSError as exc:
                    rollback_errors.append(f"{path}: {exc}")
            if rollback_errors:
                preserve_backups = True
                raise RuntimeError(
                    "artifact publication failed and rollback also failed: "
                    + "; ".join(rollback_errors)
                ) from publish_error
            raise
    finally:
        _cleanup_publish_files(
            (graph_tmp, sidecar_tmp, *(() if preserve_backups else backups.values())),
            tag,
        )
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)

    if sidecar_text is None and backups.get(sidecar_out) is not None:
        print(f"[{tag}] removed stale sidecar {sidecar_out} (source has none)")


def _engine_sidecar_plan(onnx_path: Path, out_path: Path) -> tuple[Path, Path | None]:
    """Choose an engine sidecar without overwriting the graph or ONNX metadata."""
    onnx_path = Path(onnx_path).resolve()
    out_path = Path(out_path).resolve()
    source = onnx_path.with_suffix(".json")
    protected_source = source.resolve()

    if protected_source == out_path:
        raise SystemExit(
            f"engine output would overwrite the ONNX sidecar: {out_path}; "
            "use an output ending in .engine"
        )

    candidates = [out_path.with_suffix(".json"), Path(str(out_path) + ".json")]
    usable: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved == out_path or resolved == protected_source:
            continue
        if all(resolved != existing.resolve() for existing in usable):
            usable.append(candidate)

    if not usable:
        raise SystemExit(
            f"cannot place an engine sidecar safely next to {out_path}; "
            "use an output ending in .engine"
        )

    chosen = usable[0]
    stale_twin = usable[1] if len(usable) > 1 else None
    return chosen, stale_twin


def _validated_artifact_plan(onnx_path: Path, out_path: Path) -> tuple[Path, Path | None]:
    """Validate all input, output, and staging paths before TensorRT setup."""
    onnx_path = Path(onnx_path).resolve()
    out_path = Path(out_path).resolve()
    if onnx_path.suffix.lower() != ".onnx":
        raise SystemExit(f"--onnx must end in .onnx (got {onnx_path})")
    if out_path.suffix.lower() != ".engine":
        raise SystemExit(f"--output must end in .engine (got {out_path})")

    engine_sidecar, stale_twin = _engine_sidecar_plan(onnx_path, out_path)
    paths = {
        "input ONNX": onnx_path,
        "input sidecar": onnx_path.with_suffix(".json").resolve(),
        "engine output": out_path,
        "engine staging file": Path(str(out_path) + ".tmp").resolve(),
        "engine sidecar": engine_sidecar.resolve(),
        "sidecar staging file": Path(str(engine_sidecar) + ".tmp").resolve(),
    }
    seen: dict[Path, str] = {}
    for label, path in paths.items():
        if previous := seen.get(path):
            raise SystemExit(
                f"artifact path collision: {previous} and {label} both resolve to {path}"
            )
        seen[path] = label
    return engine_sidecar, stale_twin


def _load_onnx_metadata(onnx_path: Path) -> tuple[Path, dict, bool]:
    """Load and validate metadata that affects the runtime contract."""
    sidecar = onnx_path.with_suffix(".json")
    if not sidecar.exists():
        return sidecar, {}, False
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read ONNX sidecar {sidecar}: {exc}") from exc
    if not isinstance(meta, dict):
        raise SystemExit(f"ONNX sidecar must contain a JSON object: {sidecar}")
    artifact_kind = meta.get("artifact_kind")
    if artifact_kind not in (None, "onnx"):
        raise SystemExit(
            f"source sidecar {sidecar} declares artifact_kind={artifact_kind!r}; expected 'onnx'"
        )
    color_order = meta.get("color_order", "RGB")
    if color_order != "RGB":
        raise SystemExit(
            f"source sidecar {sidecar} declares color_order={color_order!r}; "
            "D-FINE model input must be RGB; mark BGR source images at runtime"
        )
    return sidecar, meta, True


def _file_identity(path: Path) -> tuple[int, int, int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _load_onnx_artifact(onnx_path: Path) -> tuple[Path, dict, bool, bytes]:
    """Snapshot the graph and optional sidecar as one stable build input."""
    sidecar_path = onnx_path.with_suffix(".json")
    before = _file_identity(onnx_path), _file_identity(sidecar_path)
    sidecar, meta, have_source_meta = _load_onnx_metadata(onnx_path)
    onnx_bytes = onnx_path.read_bytes()
    after = _file_identity(onnx_path), _file_identity(sidecar_path)
    if before != after:
        raise SystemExit("source ONNX artifact changed while it was being read; retry the build")
    return sidecar, meta, have_source_meta, onnx_bytes


def _graph_policy(max_aux_streams: int | None, cuda_graph_alias: bool) -> dict:
    """Resolve the TensorRT auxiliary-stream policy."""
    if max_aux_streams is not None and max_aux_streams < 0:
        raise SystemExit("--max-aux-streams must be non-negative")
    if cuda_graph_alias:
        if max_aux_streams not in (None, 0):
            raise SystemExit("--cuda-graph conflicts with --max-aux-streams > 0")
        max_aux_streams = 0
    return {"max_aux_streams": max_aux_streams}


def _validate_batch_profile(min_batch: int, opt_batch: int, max_batch: int) -> None:
    """Reject invalid optimization profiles before creating TensorRT objects."""
    if min_batch < 1 or not min_batch <= opt_batch <= max_batch:
        raise SystemExit(
            "batch profile must satisfy 1 <= min <= opt <= max "
            f"(got {min_batch}/{opt_batch}/{max_batch})"
        )
    if min_batch == max_batch and min_batch != 1:
        raise SystemExit(
            "static batch profiles above 1 are not supported by the native runtime; "
            f"use 1/{opt_batch}/{max_batch} to target batch {opt_batch}"
        )


def _apply_graph_policy(config, policy: dict) -> int | None:
    """Apply the resolved auxiliary-stream limit to a TensorRT config."""
    max_aux_streams = policy["max_aux_streams"]
    if max_aux_streams is not None:
        config.max_aux_streams = max_aux_streams
    return max_aux_streams


def _graph_outputs_are_fp32(network, meta: dict) -> bool:
    outputs = [network.get_output(index) for index in range(network.num_outputs)]
    by_name = {tensor.name: tensor for tensor in outputs}
    selected = None

    names = meta.get("output_names")
    if isinstance(names, list) and len(names) == 2 and names[0] != names[1]:
        if all(name in by_name for name in names):
            selected = [by_name[name] for name in names]
    elif "logits" in by_name and "boxes" in by_name:
        selected = [by_name["logits"], by_name["boxes"]]
    elif len(outputs) == 2:
        boxes = [
            tensor for tensor in outputs if tuple(tensor.shape) and tuple(tensor.shape)[-1] == 4
        ]
        if len(boxes) == 1:
            selected = outputs

    return selected is not None and all(tensor.dtype == trt.DataType.FLOAT for tensor in selected)


def _engine_batch_facts(
    engine,
    input_name: str,
    chw: tuple[int, int, int],
    requested: tuple[int, int, int],
) -> dict:
    """Read the effective batch contract from a deserialized engine."""
    if engine.num_optimization_profiles != 1:
        raise RuntimeError(
            f"built engine has {engine.num_optimization_profiles} optimization profiles; expected 1"
        )

    shape = tuple(int(dim) for dim in engine.get_tensor_shape(input_name))
    if len(shape) != 4 or tuple(shape[1:]) != chw or shape[0] == 0 or shape[0] < -1:
        raise RuntimeError(f"built engine has invalid input shape for {input_name}: {shape}")

    profile_shapes = tuple(
        tuple(int(dim) for dim in dims) for dims in engine.get_tensor_profile_shape(input_name, 0)
    )
    if len(profile_shapes) != 3 or any(
        len(dims) != 4 or tuple(dims[1:]) != chw or dims[0] < 1 for dims in profile_shapes
    ):
        raise RuntimeError(
            f"built engine has invalid optimization profile for {input_name}: {profile_shapes}"
        )
    dynamic_batch = shape[0] == -1
    if not dynamic_batch and shape[0] != 1:
        raise RuntimeError(
            f"built engine has static batch {shape[0]}; the native runtime supports static batch 1 "
            "or a dynamic batch profile"
        )

    expected_shapes = tuple((batch, *chw) for batch in requested)
    if profile_shapes != expected_shapes:
        raise RuntimeError(
            f"built engine batch profile {profile_shapes} differs from requested {expected_shapes}"
        )

    min_batch, opt_batch, max_batch = (dims[0] for dims in profile_shapes)
    return {
        "dynamic_batch": dynamic_batch,
        "min_batch": min_batch,
        "opt_batch": opt_batch,
        "max_batch": max_batch,
    }


def _engine_build_facts(
    args: argparse.Namespace,
    onnx_sha256: str,
    graph_policy: dict,
    outputs_fp32: bool,
    batch_facts: dict,
) -> dict:
    """Return metadata owned by the TensorRT build step."""
    return {
        "artifact_kind": "engine",
        "schema_version": 1,
        "network_typing": "strong" if args.strongly_typed else "weak",
        "trt_version": trt.__version__,
        "sm_arch": _sm_arch(),
        "tf32": not args.no_tf32,
        **batch_facts,
        "onnx_sha256": onnx_sha256,
        **graph_policy,
        "cuda_graph_compat": graph_policy["max_aux_streams"] == 0 and outputs_fp32,
    }


def _publish_engine_pair(
    engine_tmp: Path,
    engine_out: Path,
    sidecar_text: str,
    sidecar_out: Path,
    stale_twin: Path | None,
) -> None:
    """Publish a complete pair, then remove an obsolete alternate sidecar."""
    stale_existed = stale_twin is not None and stale_twin.is_file()
    _publish_pair(
        engine_tmp,
        engine_out,
        sidecar_text,
        sidecar_out,
        "build",
        stale_sidecar=stale_twin,
    )
    if stale_existed:
        print(f"[build] removed stale sidecar {stale_twin.name}")


def pin_decoder_fp32(
    network: "trt.INetworkDefinition", prefixes: tuple[str, ...], verbose: bool
) -> int:
    """Apply the experimental FP32 decoder constraints to a weakly typed graph.

    Only layers with floating-point outputs are pinned; shape and integer plumbing
    under the same prefix is left unchanged. These placement modes did not recover
    the validated typed-graph accuracy and are retained for controlled comparison.
    """
    pinned = 0
    skipped = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if not layer.name.startswith(prefixes):
            continue
        float_outs = [
            j
            for j in range(layer.num_outputs)
            if layer.get_output(j).dtype in (trt.DataType.FLOAT, trt.DataType.HALF)
        ]
        if not float_outs:
            skipped += 1  # shape / int-only layer under the decoder scope
            continue
        if layer.type not in _NO_COMPUTE_PRECISION:
            layer.precision = trt.DataType.FLOAT
        for j in float_outs:
            layer.set_output_type(j, trt.DataType.FLOAT)  # keeps FP32 weights/activations
        pinned += 1
    print(
        f"[build] pinned {pinned} decoder layers to FP32 "
        f"(prefixes={list(prefixes)}, skipped {skipped} shape/int layers)"
    )
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
    planned_sidecar, stale_twin = _validated_artifact_plan(onnx_path, out_path)
    sidecar, meta, have_source_meta, onnx_bytes = _load_onnx_artifact(onnx_path)
    onnx_sha256 = hashlib.sha256(onnx_bytes).hexdigest()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _validate_batch_profile(args.min_batch, args.opt_batch, args.max_batch)

    if (
        sum(bool(x) for x in (args.fp16, args.fp16_decoder_fp32, args.bf16_decoder_fp32, args.int8))
        > 1
    ):
        raise SystemExit(
            "--fp16, --fp16-decoder-fp32, --bf16-decoder-fp32 and --int8 are mutually exclusive"
        )
    if (args.fp16_decoder_fp32 or args.bf16_decoder_fp32) and args.strongly_typed:
        # Per-layer setPrecision is rejected on a strongly-typed network (types come
        # from the ONNX). The mixed build is weakly-typed by construction.
        raise SystemExit(
            "mixed-precision pinning needs a weakly-typed network; drop --strongly-typed"
        )
    graph_policy = _graph_policy(args.max_aux_streams, args.cuda_graph)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 0
    if args.strongly_typed:
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        print("[build] strongly-typed network (precision pinned by ONNX types)")
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)

    print(f"[build] TensorRT {trt.__version__}")
    if not parser.parse(onnx_bytes):
        for i in range(parser.num_errors):
            print(f"[build][parser] {parser.get_error(i)}")
        raise RuntimeError("ONNX parse failed")
    print(
        f"[build] parsed {onnx_path.name}: "
        f"{network.num_inputs} inputs, {network.num_outputs} outputs, {network.num_layers} layers"
    )

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gb << 30)

    max_aux_streams = _apply_graph_policy(config, graph_policy)
    if max_aux_streams is not None:
        # TRT uses auxiliary streams for intra-inference parallelism by default (2 here).
        # A CUDA graph captured with cudaStreamCaptureModeThreadLocal only records the main
        # stream, silently missing aux-stream kernels -> an incomplete/incorrect graph. So a
        # graph-capturable engine must be built single-stream (--max-aux-streams 0); the
        # C++ detector gates capture on num_aux_streams()==0 exactly for this reason.
        print(
            f"[build] max_aux_streams = {max_aux_streams}"
            + ("  (single-stream, CUDA-graph capturable)" if max_aux_streams == 0 else "")
        )

    if args.no_tf32:
        # TF32 is on by default and deviates ~1% from true FP32 PyTorch (the FDR
        # integral amplifies it into box error). Disable it for an FP32-faithful
        # parity reference; leave it on for production speed.
        config.clear_flag(trt.BuilderFlag.TF32)
        print("[build] TF32 disabled (FP32-faithful build)")

    if args.tactic:
        names = {
            "cublas": trt.TacticSource.CUBLAS,
            "cublaslt": trt.TacticSource.CUBLAS_LT,
            "cudnn": trt.TacticSource.CUDNN,
            "edge": trt.TacticSource.EDGE_MASK_CONVOLUTIONS,
            "jit": trt.TacticSource.JIT_CONVOLUTIONS,
        }
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
    profile.set_shape(
        inp.name, (args.min_batch, c, h, w), (args.opt_batch, c, h, w), (args.max_batch, c, h, w)
    )
    config.add_optimization_profile(profile)
    print(
        f"[build] profile {inp.name}: min={args.min_batch} opt={args.opt_batch} "
        f"max={args.max_batch} (CHW={c}x{h}x{w})"
    )

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
        constraint = (
            trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS
            if args.constraints == "obey"
            else trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS
        )
        config.set_flag(constraint)
        prefixes = tuple(p for p in args.decoder_prefixes.split(",") if p)
        n = pin_decoder_fp32(network, prefixes, args.verbose)
        if n == 0:
            raise RuntimeError(
                f"no layers matched prefixes {list(prefixes)} — "
                "check the ONNX layer naming and precision recipe"
            )
        print(
            f"[build] {low_name} mixed: unpinned layers {low_name}, {n} pinned FP32 "
            f"({args.constraints.upper()}_PRECISION_CONSTRAINTS)"
        )

    outputs_fp32 = _graph_outputs_are_fp32(network, meta)
    input_name = inp.name
    chw = (int(c), int(h), int(w))
    input_shape = tuple(inp.shape)
    output_shapes = [tuple(network.get_output(i).shape) for i in range(network.num_outputs)]
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("build_serialized_network returned None")
    del profile, inp, parser, network, config, builder
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        raise RuntimeError("failed to deserialize the engine built in this process")
    batch_facts = _engine_batch_facts(
        engine,
        input_name,
        chw,
        (args.min_batch, args.opt_batch, args.max_batch),
    )
    batch_kind = "dynamic" if batch_facts["dynamic_batch"] else "static"
    print(
        f"[build] verified {batch_kind} engine batch profile: "
        f"min={batch_facts['min_batch']} opt={batch_facts['opt_batch']} "
        f"max={batch_facts['max_batch']}"
    )
    del engine, runtime
    # Engine sidecar: the ONNX contract passes through untouched; the builder only
    # appends facts IT owns. Precision is decided by whoever set the compute types:
    # a weakly-typed flag mode here, or the converter's ONNX types (strongly typed) —
    # the builder must never overwrite the converter's recipe with a flag guess
    # (v0.3.0 stamped every strongly-typed FP16 engine "fp32" this way).
    if not have_source_meta:
        # No contract sidecar: record what the parsed network itself asserts, so
        # the runtime's sidecar-vs-engine cross-check sees the engine's real
        # dims/classes/queries instead of absent fields (class names and
        # normalization stay unknown — the runtime warns and uses its defaults).
        logits_shape = next((s for s in output_shapes if len(s) == 3 and s[-1] != 4), None)
        if len(input_shape) == 4 and input_shape[2] > 0 and input_shape[3] > 0:
            meta["input_h"], meta["input_w"] = int(input_shape[2]), int(input_shape[3])
        if logits_shape and logits_shape[1] > 0 and logits_shape[2] > 0:
            meta["num_queries"], meta["num_classes"] = int(logits_shape[1]), int(logits_shape[2])
        print(
            f"[build] NOTE: no ONNX sidecar ({sidecar.name}) — engine sidecar carries the "
            "graph contract + build facts only; preprocessing uses runtime defaults"
        )
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
            meta.setdefault("precision", "fp32" if have_source_meta else "unknown")
            meta.setdefault(
                "precision_mode",
                "strongly_typed_unknown" if meta["precision"] != "fp32" else "fp32",
            )
        else:
            meta.setdefault("precision", "fp32")
            meta.setdefault(
                "precision_mode",
                "fp32" if meta["precision"] == "fp32" else "strongly_typed_unknown",
            )
    meta.update(_engine_build_facts(args, onnx_sha256, graph_policy, outputs_fp32, batch_facts))
    # Revalidate the resolved destinations after the build so a path change cannot
    # redirect publication. The ONNX sidecar namespace remains reserved whether or
    # not a source sidecar existed in the input snapshot.
    engine_sidecar, current_stale_twin = _validated_artifact_plan(onnx_path, out_path)
    if (engine_sidecar, current_stale_twin) != (planned_sidecar, stale_twin):
        raise RuntimeError("engine sidecar destination changed during the build")

    # Stage the engine only after all destination checks have passed. Publishing
    # the staged sidecar and engine uses adjacent atomic renames.
    tmp_path = _adjacent_temp(out_path)
    try:
        tmp_path.write_bytes(plan)
        _publish_engine_pair(
            tmp_path,
            out_path,
            json.dumps(meta, indent=2) + "\n",
            engine_sidecar,
            stale_twin,
        )
    finally:
        tmp_path.unlink(missing_ok=True)
    print(f"[build] wrote {out_path} ({plan.nbytes / 1e6:.1f} MB)")
    print(
        f"[build] wrote engine sidecar {engine_sidecar} "
        f"(precision={meta['precision']}, mode={meta['precision_mode']})"
    )


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Build a TensorRT engine from a raw D-FINE ONNX")
    p.add_argument("--onnx", default=str(repo / "trt-files" / "onnx" / "dfine_m.onnx"))
    p.add_argument("--output", default=str(repo / "trt-files" / "engines" / "dfine_m_fp32.engine"))
    p.add_argument("--min-batch", type=int, default=1)
    p.add_argument("--opt-batch", type=int, default=1)
    p.add_argument("--max-batch", type=int, default=8)
    p.add_argument("--workspace-gb", type=int, default=4)
    p.add_argument(
        "--fp16",
        action="store_true",
        help="naive whole-graph FP16 (anti-example: corrupts the decoder, timing only)",
    )
    p.add_argument(
        "--fp16-decoder-fp32",
        action="store_true",
        help="mixed FP16: unpinned layers FP16, --decoder-prefixes layers pinned FP32",
    )
    p.add_argument(
        "--bf16-decoder-fp32",
        action="store_true",
        help="mixed BF16 (no overflow, less mantissa): unpinned BF16, pinned FP32",
    )
    p.add_argument(
        "--int8",
        action="store_true",
        help="INT8 from an explicit-QDQ ONNX (convert_int8.py); decoder stays FP32",
    )
    p.add_argument(
        "--decoder-prefixes",
        default="/model/decoder,model.decoder",
        help="comma-separated layer-name prefixes identifying the decoder to pin FP32",
    )
    p.add_argument(
        "--constraints",
        choices=["obey", "prefer"],
        default="obey",
        help="how strictly TRT honours the FP32 pins (--fp16-decoder-fp32)",
    )
    p.add_argument("--no-tf32", action="store_true", help="disable TF32 for an FP32-faithful build")
    p.add_argument("--prefer-precision", action="store_true", help="PREFER_PRECISION_CONSTRAINTS")
    p.add_argument("--opt-level", type=int, default=None, help="builder_optimization_level 0-5")
    p.add_argument(
        "--strongly-typed",
        action="store_true",
        help=(
            "derive TensorRT computation types from ONNX tensor types; required for typed FP16 "
            "artifacts"
        ),
    )
    p.add_argument(
        "--tactic", default=None, help="restrict tactic sources, e.g. 'cublas' or 'cublas,edge,jit'"
    )
    p.add_argument(
        "--cuda-graph", action="store_true", help="set --max-aux-streams 0 for CUDA Graph capture"
    )
    p.add_argument(
        "--max-aux-streams",
        type=int,
        default=None,
        help="cap TRT auxiliary streams; 0 = single-stream (required for CUDA-graph capture)",
    )
    p.add_argument("--verbose", action="store_true", help="list every FP32-pinned decoder layer")
    return p.parse_args()


if __name__ == "__main__":
    build(parse_args())
