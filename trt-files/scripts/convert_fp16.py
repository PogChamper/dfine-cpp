#!/usr/bin/env python3
"""Convert a raw D-FINE ONNX to mixed FP16 (backbone+encoder FP16, decoder FP32).

This is the *strongly-typed* path to FP16, and it exists because the weakly-typed
`config.set_flag(kFP16)` route degrades D-FINE by ~6.8 AP even with every compute
layer pinned FP32 — TRT inserts uncontrolled FP16 reformats on the FDR's
precision-critical data path. Here the precision is baked into
the ONNX tensor types instead: onnxconverter_common casts backbone+encoder tensors to
FP16 and inserts explicit Cast nodes at the decoder boundary, and the decoder
(block-listed by name) stays FP32. Build the result with `build_engine.py
--strongly-typed` (NO kFP16 flag), so TRT reproduces exactly these types.

    convert_fp16.py --onnx dfine_m.onnx --output dfine_m_fp16_st.onnx
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import secrets
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper
from onnxconverter_common import float16


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
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    if source_path.suffix.lower() != ".onnx":
        raise SystemExit(f"[fp16] --onnx must end in .onnx (got {source_path})")
    if output_path.suffix.lower() != ".onnx":
        raise SystemExit(f"[fp16] --output must end in .onnx (got {output_path})")
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
                f"[fp16] artifact path collision: {previous} and {label} resolve to {path}"
            )
        seen[path] = label
    return source_path, output_path


def _file_identity(path: Path) -> tuple[int, int, int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _load_source_artifact(source_path: Path) -> tuple[onnx.ModelProto, dict | None]:
    sidecar = source_path.with_suffix(".json")
    before = _file_identity(source_path), _file_identity(sidecar)
    source_bytes = source_path.read_bytes()
    sidecar_bytes = sidecar.read_bytes() if before[1] is not None else None
    after = _file_identity(source_path), _file_identity(sidecar)
    if before != after:
        raise SystemExit("[fp16] source ONNX artifact changed while it was being read; retry")
    model = onnx.load_model_from_string(source_bytes)
    if sidecar_bytes is None:
        return model, None
    try:
        meta = json.loads(sidecar_bytes)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[fp16] cannot parse source sidecar {sidecar}: {exc}") from exc
    if not isinstance(meta, dict):
        raise SystemExit("[fp16] source sidecar must contain a JSON object")
    return model, meta


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
    vtype = {
        vi.name: vi.type.tensor_type.elem_type
        for vi in list(g.value_info) + list(g.input) + list(g.output)
    }
    inits = {i.name: i for i in g.initializer}

    def elem_type(name):
        return inits[name].data_type if name in inits else vtype.get(name)

    shared = {"Mul", "Add", "Sub", "Div", "Pow", "Min", "Max", "MatMul", "Gemm", "Where"}
    fixed = 0
    dup = {}  # (init_name, target) -> duplicated init name
    pending_casts = []  # (node_index, cast_node) — inserted before the consumer (topo order)
    for ni, node in enumerate(g.node):
        if node.op_type not in shared:
            continue
        types = {elem_type(i) for i in node.input if elem_type(i) in F}
        if len(types) < 2:
            continue
        target = (
            TensorProto.FLOAT if node.name.startswith(decoder_prefixes) else TensorProto.FLOAT16
        )
        npd = np.float16 if target == TensorProto.FLOAT16 else np.float32
        for idx, iname in enumerate(node.input):
            if elem_type(iname) not in F or elem_type(iname) == target:
                continue
            if iname in inits:  # duplicate the constant as a target-typed copy for this node
                key = (iname, target)
                if key not in dup:
                    newname = f"{iname}__{'f16' if target == TensorProto.FLOAT16 else 'f32'}"
                    g.initializer.append(
                        numpy_helper.from_array(
                            numpy_helper.to_array(inits[iname]).astype(npd), newname
                        )
                    )
                    inits[newname] = g.initializer[-1]
                    dup[key] = newname
                node.input[idx] = dup[key]
            else:  # activation: insert a boundary Cast
                cast_out = f"{iname}__harm{fixed}"
                pending_casts.append(
                    (
                        ni,
                        onnx.helper.make_node(
                            "Cast", [iname], [cast_out], to=target, name=f"harmonize_cast_{fixed}"
                        ),
                    )
                )
                node.input[idx] = cast_out
                vtype[cast_out] = target
            fixed += 1
    for ni, cast in sorted(pending_casts, key=lambda x: -x[0]):
        g.node.insert(ni, cast)
    return fixed


def main(args: argparse.Namespace) -> None:
    onnx_path, out_path = _validated_conversion_paths(args.onnx, args.output)
    model, source_meta = _load_source_artifact(onnx_path)

    prefixes = tuple(p for p in args.fp32_prefixes.split(",") if p)
    block = keep_fp32_nodes(model, prefixes)
    print(
        f"[fp16] keeping {len(block)} nodes FP32 (prefixes={list(prefixes)}); converting the rest to FP16"
    )

    # keep_io_types keeps the input FP32 but still appends a trailing Cast-to-FP16 on
    # each graph output. The decoder (block-listed) already emits FP32, so retype the
    # outputs back to FP32 — profile.py's trt backend and CUDA-graph replay want FP32
    # outputs, and it just drops a redundant downcast.
    model16 = float16.convert_float_to_float16(
        model, node_block_list=block, keep_io_types=True, disable_shape_infer=False
    )

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
            model16.graph.node.append(
                helper.make_node("Cast", [inner], [out.name], to=TensorProto.FLOAT)
            )
        out.type.tensor_type.elem_type = TensorProto.FLOAT
    n_harmonized = harmonize_float_types(model16, prefixes)
    if n_harmonized:
        print(
            f"[fp16] harmonized {n_harmonized} mixed Half/Float node inputs "
            "(strongly-typed TRT would reject them)"
        )
    onnx.checker.check_model(model16)
    tmp = _adjacent_temp(out_path)
    sidecar_text = None
    try:
        onnx.save(model16, str(tmp))
        if source_meta is not None:
            meta = dict(source_meta)
            meta["precision"] = "fp16"
            meta["fp16_decoder_fp32"] = True
            meta["precision_mode"] = "strongly_typed_onnx_fp16"
            sidecar_text = json.dumps(meta, indent=2) + "\n"
        _publish_pair(tmp, out_path, sidecar_text, out_path.with_suffix(".json"), "fp16")
    finally:
        tmp.unlink(missing_ok=True)

    n_cast = sum(1 for n in model16.graph.node if n.op_type == "Cast")
    print(f"[fp16] wrote {out_path} ({n_cast} Cast nodes at precision boundaries)")
    if sidecar_text is not None:
        print(f"[fp16] wrote sidecar {out_path.with_suffix('.json')}")


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(
        description="ONNX-level mixed FP16 (decoder kept FP32) for strong typing"
    )
    p.add_argument("--onnx", default=str(repo / "trt-files" / "onnx" / "dfine_m.onnx"))
    p.add_argument("--output", default=str(repo / "trt-files" / "onnx" / "dfine_m_fp16_st.onnx"))
    p.add_argument(
        "--fp32-prefixes",
        default="/model/decoder,model.decoder",
        help="node-name prefixes to keep FP32 (default: the whole decoder)",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
