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
import fcntl
import hashlib
import importlib.metadata
import json
import os
import secrets
from pathlib import Path

import onnx
from onnx import TensorProto, helper
from onnxconverter_common import float16

FLOAT_TYPES = (TensorProto.FLOAT, TensorProto.FLOAT16)
STOP_OPS = {"MatMul", "Gemm", "Softmax", "Conv"}
_EXTRA = tuple(x for x in os.environ.get("SURGICAL_EXTRA_SCOPES", "").split(",") if x)
FDR_SCOPES = _EXTRA + (
    "/model/decoder/decoder/integral",
    "/model/decoder/decoder/lqe",
    "dec_bbox_head",
    "/model/decoder/pre_bbox_head",
    "/model/decoder/enc_bbox_head",
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
            (graph_tmp, sidecar_tmp, *(() if preserve_backups else backups.values())), tag
        )
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)

    if sidecar_text is None and backups.get(sidecar_out) is not None:
        print(f"[{tag}] removed stale sidecar {sidecar_out} (source has none)")


def _validated_conversion_paths(source: str | Path, output: str | Path) -> tuple[Path, Path]:
    """Reserve distinct graph, sidecar, and staging paths."""
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    if source_path.suffix.lower() != ".onnx":
        raise SystemExit(f"[surgical] --onnx must end in .onnx (got {source_path})")
    if output_path.suffix.lower() != ".onnx":
        raise SystemExit(f"[surgical] --output must end in .onnx (got {output_path})")
    paths = {
        "source ONNX": source_path,
        "source sidecar": source_path.with_suffix(".json").resolve(),
        "output ONNX": output_path,
        "output staging file": Path(str(output_path) + ".tmp").resolve(),
        "output sidecar": output_path.with_suffix(".json").resolve(),
        "sidecar staging file": Path(str(output_path.with_suffix(".json")) + ".tmp").resolve(),
    }
    seen: dict[Path, str] = {}
    for label, path in paths.items():
        if previous := seen.get(path):
            raise SystemExit(
                f"[surgical] artifact path collision: {previous} and {label} resolve to {path}"
            )
        seen[path] = label
    return source_path, output_path


def _conversion_overrides() -> dict:
    overrides = {}
    if _EXTRA:
        overrides["extra_fp32_scopes"] = list(_EXTRA)
    fp16_only = tuple(x for x in os.environ.get("SURGICAL_FP16_ONLY", "").split(",") if x)
    if fp16_only:
        overrides["fp16_only_scopes"] = list(fp16_only)
    return overrides


def _converted_sidecar(
    meta: dict,
    source_bytes: bytes,
    slim: bool,
    overrides: dict | None = None,
) -> dict:
    """Carry the export contract forward and fingerprint this conversion."""
    tool_versions = meta.get("tool_versions", {})
    if not isinstance(tool_versions, dict):
        raise SystemExit("[surgical] source sidecar tool_versions must be an object")
    converted = dict(meta)
    overrides = overrides or {}
    converted.update(
        precision="fp16",
        precision_mode=(
            "strongly_typed_onnx_fp16_surgical_experimental"
            if overrides
            else (
                "strongly_typed_onnx_fp16_surgical_slim"
                if slim
                else "strongly_typed_onnx_fp16_surgical_decoder"
            )
        ),
        source_onnx_sha256=hashlib.sha256(source_bytes).hexdigest(),
        converter_sha256=hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
        tool_versions={
            **tool_versions,
            "onnxconverter-common": importlib.metadata.version("onnxconverter-common"),
        },
    )
    if overrides:
        converted["conversion_overrides"] = overrides
    else:
        converted.pop("conversion_overrides", None)
    return converted


def _load_source_onnx(source_path: Path) -> tuple[onnx.ModelProto, bytes]:
    """Load and fingerprint one immutable snapshot of the source graph."""
    source_bytes = source_path.read_bytes()
    return onnx.load_model_from_string(source_bytes), source_bytes


def _file_identity(path: Path) -> tuple[int, int, int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _load_source_artifact(source_path: Path) -> tuple[onnx.ModelProto, bytes, dict | None]:
    """Snapshot a source graph and sidecar before conversion starts."""
    sidecar = source_path.with_suffix(".json")
    before = _file_identity(source_path), _file_identity(sidecar)
    model, source_bytes = _load_source_onnx(source_path)
    sidecar_bytes = sidecar.read_bytes() if before[1] is not None else None
    after = _file_identity(source_path), _file_identity(sidecar)
    if before != after:
        raise SystemExit("[surgical] source ONNX artifact changed while it was being read; retry")
    if sidecar_bytes is None:
        return model, source_bytes, None
    try:
        meta = json.loads(sidecar_bytes)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[surgical] cannot parse source sidecar {sidecar}: {exc}") from exc
    if not isinstance(meta, dict):
        raise SystemExit("[surgical] source sidecar must contain a JSON object")
    return model, source_bytes, meta


def is_glue_leaf(name: str) -> bool:
    """Leaf op directly under /model/decoder/ or /model/decoder/decoder/."""
    for prefix in ("/model/decoder/decoder/", "/model/decoder/"):
        if name.startswith(prefix):
            rest = name[len(prefix) :]
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
            if ("/model/decoder" in n.name or "model.decoder" in n.name) and not any(
                s in n.name for s in fp16_only
            ):
                block.add(n.name)
                n_scope += 1
        n_coord = 0
        if any("cross_attn" in s for s in fp16_only):
            # hybrid: cross_attn data path fp16, coordinate/index math re-blocked FP32
            cs = coord_slice(g, by_output)
            n_coord = len(cs - block)
            block |= cs
        print(
            f"[surgical] coarse mode: decoder FP32 except {fp16_only}"
            + (f" (+{n_coord} coord-slice nodes re-blocked FP32)" if n_coord else "")
        )
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

    print(
        f"[surgical] blocklist{' (slim: glue leaves stay FP16)' if slim else ''}: "
        f"{n_scope} FDR-scope + {n_glue} glue-leaf + "
        f"{n_coord} deform-coordinate nodes = {len(block)} total "
        f"(of {len(g.node)} graph nodes)"
    )
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
            g.node.append(
                helper.make_node(
                    "Cast", [inner], [out.name], name=out.name + "_to_fp32", to=TensorProto.FLOAT
                )
            )
        out.type.tensor_type.elem_type = TensorProto.FLOAT


# ops whose outputs are never float regardless of inputs
NONFLOAT_OUT = {
    "Shape",
    "ArgMax",
    "ArgMin",
    "Equal",
    "Less",
    "Greater",
    "LessOrEqual",
    "GreaterOrEqual",
    "And",
    "Or",
    "Not",
    "NonZero",
}


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
        float_ins = [
            (i, vtype.get(x))
            for i, x in enumerate(n.input)
            if i not in skip and vtype.get(x) in FLOAT_TYPES
        ]
        target = TensorProto.FLOAT if n.name in block else TensorProto.FLOAT16
        for i, t in float_ins:
            if t == target:
                continue
            iname = n.input[i]
            if iname in inits:
                key = (iname, target)
                if key not in dup_cache:
                    arr = onnx.numpy_helper.to_array(inits[iname])
                    arr = arr.astype(np.float16 if target == TensorProto.FLOAT16 else np.float32)
                    suffix = "__f16" if target == TensorProto.FLOAT16 else "__f32"
                    ni = onnx.numpy_helper.from_array(arr, iname + suffix)
                    g.initializer.append(ni)
                    vtype[iname + suffix] = target
                    dup_cache[key] = iname + suffix
                n.input[i] = dup_cache[key]
            else:
                cast_out = f"{iname}__harm{fixed}"
                cast = helper.make_node(
                    "Cast", [iname], [cast_out], name=f"harmonize_cast_{fixed}", to=target
                )
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
    p.add_argument(
        "--slim",
        action="store_true",
        help="leave the decoder glue leaves FP16 too (FP32 island = FDR scopes "
        "+ deform coordinate slice only) — measured lossless on COCO "
        "full-val for all five sizes, +2-3%% b8; the release default",
    )
    args = p.parse_args()
    env_slim = os.environ.get("SURGICAL_NO_GLUE", "").strip().lower()
    slim = args.slim or env_slim not in ("", "0", "false", "no", "off")

    source_path, output_path = _validated_conversion_paths(args.onnx, args.output)
    model, source_bytes, source_meta = _load_source_artifact(source_path)
    opset = max(
        (imp.version for imp in model.opset_import if imp.domain in ("", "ai.onnx")), default=0
    )
    if opset < 19 and not os.environ.get("SURGICAL_FP16_ONLY"):
        raise SystemExit(
            f"[surgical] input opset is {opset}, need >= 19: opset-16 exports decompose "
            "LayerNorm and TensorRT miscompiles the decomposition in FP16 (mAP ~0.005). "
            "Re-export with export_dfine_onnx.py --opset 19."
        )
    overrides = _conversion_overrides()
    block = build_blocklist(model, slim=slim)
    model16 = float16.convert_float_to_float16(
        model, node_block_list=block, keep_io_types=True, disable_shape_infer=False
    )
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
    tmp = _adjacent_temp(output_path)
    sidecar_text = None
    try:
        onnx.save(model16, tmp)
        if source_meta is not None:
            meta = _converted_sidecar(source_meta, source_bytes, slim, overrides)
            sidecar_text = json.dumps(meta, indent=2) + "\n"
        _publish_pair(tmp, output_path, sidecar_text, output_path.with_suffix(".json"), "surgical")
    finally:
        tmp.unlink(missing_ok=True)
    print(f"[surgical] wrote {output_path}")


if __name__ == "__main__":
    main()
