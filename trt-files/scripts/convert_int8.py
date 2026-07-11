#!/usr/bin/env python3
"""Insert INT8 QDQ nodes into a raw D-FINE ONNX over the backbone+encoder only.

Explicit (QDQ) quantization, NOT TensorRT's deprecated implicit IInt8EntropyCalibrator2:
onnxruntime.quantization.quantize_static places QuantizeLinear/DequantizeLinear pairs
around Conv/MatMul, with scales calibrated on real COCO images. The decoder is excluded
by name (all its nodes are cleanly scoped under /model/decoder + model.decoder), so it
carries no Q/DQ and TensorRT runs it in FP32 — the FP-sensitive FDR path stays faithful,
exactly as in the FP16 mixed build. Build the result with `build_engine.py --int8`.

    convert_int8.py --onnx dfine_m.onnx --output dfine_m_int8_qdq.onnx --num-calib 200

Expect some mAP loss vs FP32/FP16 (INT8 on a detection transformer is lossy); quantify
it with profile.py before trusting the engine.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import secrets
import sys
from pathlib import Path

import numpy as np
import onnx

sys.path[:] = [p for p in sys.path if p not in ("", str(Path(__file__).resolve().parent))]
sys.path.append(str(Path(__file__).resolve().parent))
import cv2  # noqa: E402
from coco_eval import preprocess  # noqa: E402  (the exact D-FINE /255 stretch pipeline)
from onnxruntime.quantization import (  # noqa: E402
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process  # noqa: E402


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
        raise SystemExit(f"[int8] --onnx must end in .onnx (got {source_path})")
    if output_path.suffix.lower() != ".onnx":
        raise SystemExit(f"[int8] --output must end in .onnx (got {output_path})")
    preprocessed = output_path.with_suffix(".preproc.onnx").resolve()
    source_snapshot = Path(str(output_path) + ".source.tmp.onnx").resolve()
    paths = {
        "source ONNX": source_path,
        "source sidecar": source_path.with_suffix(".json").resolve(),
        "preprocessed ONNX": preprocessed,
        "source snapshot": source_snapshot,
        "output ONNX": output_path,
        "output staging file": Path(str(output_path) + ".tmp").resolve(),
        "output sidecar": output_path.with_suffix(".json").resolve(),
        "sidecar staging file": Path(str(output_path.with_suffix(".json")) + ".tmp").resolve(),
    }
    seen: dict[Path, str] = {}
    for label, path in paths.items():
        if previous := seen.get(path):
            raise SystemExit(
                f"[int8] artifact path collision: {previous} and {label} resolve to {path}"
            )
        seen[path] = label
    return source_path, output_path


def _file_identity(path: Path) -> tuple[int, int, int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _load_source_artifact(source_path: Path) -> tuple[onnx.ModelProto, bytes, dict | None]:
    sidecar = source_path.with_suffix(".json")
    before = _file_identity(source_path), _file_identity(sidecar)
    source_bytes = source_path.read_bytes()
    sidecar_bytes = sidecar.read_bytes() if before[1] is not None else None
    after = _file_identity(source_path), _file_identity(sidecar)
    if before != after:
        raise SystemExit("[int8] source ONNX artifact changed while it was being read; retry")
    model = onnx.load_model_from_string(source_bytes)
    if sidecar_bytes is None:
        return model, source_bytes, None
    try:
        meta = json.loads(sidecar_bytes)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[int8] cannot parse source sidecar {sidecar}: {exc}") from exc
    if not isinstance(meta, dict):
        raise SystemExit("[int8] source sidecar must contain a JSON object")
    return model, source_bytes, meta


def decoder_node_names(model: onnx.ModelProto, prefixes: tuple[str, ...]) -> list[str]:
    """Every node whose name is under the decoder scope — these stay FP32 (no Q/DQ)."""
    return [n.name for n in model.graph.node if n.name.startswith(prefixes)]


class CocoCalib(CalibrationDataReader):
    """Feeds preprocessed COCO images as the 'images' input for scale calibration."""

    def __init__(self, images_dir: Path, file_names: list[str], img_size: int, input_name: str):
        self.images_dir = images_dir
        self.file_names = file_names
        self.img_size = img_size
        self.input_name = input_name
        self._it = iter(file_names)

    def get_next(self):
        for fname in self._it:
            bgr = cv2.imread(str(self.images_dir / fname), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            x = preprocess(bgr, self.img_size).numpy().astype(np.float32)  # [1,3,H,W]
            return {self.input_name: x}
        return None

    def rewind(self):
        self._it = iter(self.file_names)


def main(args: argparse.Namespace) -> None:
    onnx_path, out_path = _validated_conversion_paths(args.onnx, args.output)
    model, source_bytes, source_meta = _load_source_artifact(onnx_path)
    input_name = model.graph.input[0].name

    prefixes = tuple(p for p in args.decoder_prefixes.split(",") if p)
    exclude = decoder_node_names(model, prefixes)
    print(f"[int8] excluding {len(exclude)} decoder nodes from quantization (stay FP32)")

    # Pick calibration images (deterministic first-N of the sorted annotation).
    from pycocotools.coco import COCO

    coco = COCO(args.ann)
    img_ids = sorted(coco.getImgIds())[: args.num_calib]
    file_names = [coco.loadImgs(i)[0]["file_name"] for i in img_ids]
    print(f"[int8] calibrating on {len(file_names)} images from {args.images}")

    # ORT wants a shape-inferred, cleaned model before static quantization.
    work_paths: list[Path] = []
    try:
        source_snapshot = _adjacent_temp(out_path, ".source.onnx")
        work_paths.append(source_snapshot)
        pre_path = _adjacent_temp(out_path, ".preproc.onnx")
        work_paths.append(pre_path)
        quant_tmp = _adjacent_temp(out_path, ".quant.onnx")
        work_paths.append(quant_tmp)
        source_snapshot.write_bytes(source_bytes)
        quant_pre_process(str(source_snapshot), str(pre_path), skip_symbolic_shape=True)
        reader = CocoCalib(Path(args.images), file_names, args.img_size, input_name)
        method = {
            "minmax": CalibrationMethod.MinMax,
            "entropy": CalibrationMethod.Entropy,
            "percentile": CalibrationMethod.Percentile,
        }[args.calib_method]

        quantize_static(
            str(pre_path),
            str(quant_tmp),
            reader,
            quant_format=QuantFormat.QDQ,
            op_types_to_quantize=["Conv", "MatMul"],
            nodes_to_exclude=exclude,
            per_channel=True,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=method,
            # TensorRT requires symmetric QDQ and folds bias into the INT8 convolution.
            extra_options={
                "ActivationSymmetric": True,
                "WeightSymmetric": True,
                "QuantizeBias": False,
            },
        )

        q = onnx.load(str(quant_tmp))
        n_q = sum(1 for node in q.graph.node if node.op_type == "QuantizeLinear")
        n_dq = sum(1 for node in q.graph.node if node.op_type == "DequantizeLinear")

        sidecar_text = None
        if source_meta is not None:
            meta = dict(source_meta)
            meta["precision"] = "int8"
            meta["quant"] = "qdq_backbone_encoder"
            sidecar_text = json.dumps(meta, indent=2) + "\n"
        _publish_pair(quant_tmp, out_path, sidecar_text, out_path.with_suffix(".json"), "int8")
    finally:
        for path in work_paths:
            path.unlink(missing_ok=True)

    print(f"[int8] wrote {out_path}: {n_q} QuantizeLinear / {n_dq} DequantizeLinear nodes")
    if sidecar_text is not None:
        print(f"[int8] wrote sidecar {out_path.with_suffix('.json')}")


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(
        description="Insert INT8 QDQ into a D-FINE ONNX (backbone/encoder only)"
    )
    p.add_argument("--onnx", default=str(repo / "trt-files" / "onnx" / "dfine_m.onnx"))
    p.add_argument("--output", default=str(repo / "trt-files" / "onnx" / "dfine_m_int8_qdq.onnx"))
    p.add_argument("--images", default="/mnt/d/datasets/coco/val2017")
    p.add_argument("--ann", default="/mnt/d/datasets/coco/annotations/instances_val2017.json")
    p.add_argument("--num-calib", type=int, default=200)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--calib-method", choices=["minmax", "entropy", "percentile"], default="minmax")
    p.add_argument("--decoder-prefixes", default="/model/decoder,model.decoder")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
