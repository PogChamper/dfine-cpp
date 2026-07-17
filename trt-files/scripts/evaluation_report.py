"""Shared provenance contract for COCO evaluation reports."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import tempfile
from pathlib import Path

IMAGE_MANIFEST_SCHEME = (
    "SHA-256 of newline-delimited canonical JSON records containing image_id, "
    "file_name, byte size, and content SHA-256; records sorted by image_id"
)
PACKAGE_ALIASES = {
    "cuda-runtime": ("nvidia-cuda-runtime-cu12", "nvidia-cuda-runtime-cu13"),
    "tensorrt": ("tensorrt", "tensorrt-cu12", "tensorrt-cu13"),
    "onnxruntime-gpu": ("onnxruntime-gpu", "onnxruntime"),
}


def package_version(package: str) -> str | None:
    for candidate in PACKAGE_ALIASES.get(package, (package,)):
        try:
            return importlib.metadata.version(candidate)
        except importlib.metadata.PackageNotFoundError:
            pass
    return None


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_image_manifest(coco, image_ids, image_dir: str | Path) -> dict:
    ids = [int(image_id) for image_id in image_ids]
    if ids != sorted(ids) or len(ids) != len(set(ids)) or not ids:
        raise ValueError("selected image ids must be unique, non-empty, and sorted")
    root = Path(image_dir)
    digest = hashlib.sha256()
    for image_id in ids:
        info = coco.loadImgs(image_id)[0]
        path = root / info["file_name"]
        record = {
            "bytes": path.stat().st_size,
            "file_name": info["file_name"],
            "image_id": image_id,
            "sha256": sha256_file(path),
        }
        digest.update(json.dumps(record, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return {
        "image_count": len(ids),
        "image_manifest_sha256": digest.hexdigest(),
        "image_manifest_scheme": IMAGE_MANIFEST_SCHEME,
    }


def model_space_geometry(model_hw: tuple[int, int]) -> dict:
    # Reports carry the maintained stretch contract only; every producer either
    # hardcodes stretch or rejects --report for letterbox runs.
    height, width = model_hw
    if any(type(value) is not int or value <= 0 for value in (height, width)):
        raise ValueError("model dimensions must be positive integers")
    return {"input_h": height, "input_w": width, "resize": "stretch"}


def evaluation_contract(
    coco,
    image_ids,
    image_dir: str | Path,
    annotations: str | Path,
    *,
    score_threshold: float,
    topk: int,
    inference_batch_size: int,
    model_hw: tuple[int, int],
    dataset_name: str = "",
    dataset_split: str = "",
    dataset_version: str = "",
    metrics_source: str | Path,
) -> dict:
    if (
        isinstance(score_threshold, bool)
        or not isinstance(score_threshold, (int, float))
        or not math.isfinite(score_threshold)
        or not 0.0 <= score_threshold <= 1.0
    ):
        raise ValueError("score threshold must be in [0, 1]")
    if type(topk) is not int or topk <= 0:
        raise ValueError("Top-K must be a positive integer")
    if type(inference_batch_size) is not int or inference_batch_size <= 0:
        raise ValueError("inference batch size must be a positive integer")
    info = coco.dataset.get("info", {})
    description = str(info.get("description", ""))
    if not dataset_name:
        dataset_name = str(info.get("dataset", ""))
        if not dataset_name and description.lower().startswith("coco"):
            dataset_name = "COCO"
        dataset_name = dataset_name or Path(annotations).parent.name
    if not dataset_split:
        dataset_split = str(info.get("split", ""))
        if not dataset_split:
            dataset_split = Path(annotations).stem.removeprefix("instances_")
    if not dataset_version:
        source_hash = info.get("source_archive_sha256")
        if source_hash:
            dataset_version = f"source-archive-sha256:{source_hash}"
        elif info.get("year"):
            dataset_version = str(info["year"])
        else:
            dataset_version = str(info.get("version") or sha256_file(annotations))
    if any(
        not isinstance(value, str) or not value.strip()
        for value in (dataset_name, dataset_split, dataset_version)
    ):
        raise ValueError("dataset name, split, and version must be non-empty strings")
    evaluator_version = importlib.metadata.version("pycocotools")
    return {
        "schema_version": 1,
        "dataset": {
            "name": dataset_name,
            "split": dataset_split,
            "version": dataset_version,
        },
        "evaluator": {
            "name": "pycocotools.COCOeval:bbox",
            "version": evaluator_version,
            "metrics_source_sha256": sha256_file(metrics_source),
        },
        "annotations_sha256": sha256_file(annotations),
        "selection": selected_image_manifest(coco, image_ids, image_dir),
        "score_threshold": score_threshold,
        "topk": topk,
        "inference_batch_size": inference_batch_size,
        "geometry": {
            "canonical": "coco_original_pixels",
            "model_space_area": model_space_geometry(model_hw),
        },
    }


def artifact(
    kind: str,
    path: str | Path,
    *,
    recipe: str,
    runtime: str,
    sidecar: str | Path | None = None,
) -> dict:
    source = Path(path).resolve()
    if (
        any(not isinstance(value, str) or not value.strip() for value in (kind, recipe, runtime))
        or not source.is_file()
    ):
        raise ValueError(f"invalid {kind or 'unnamed'} artifact: {source}")
    result = {
        "kind": kind,
        "path": str(source),
        "sha256": sha256_file(source),
        "recipe": recipe,
        "runtime": runtime,
    }
    if sidecar:
        metadata = Path(sidecar).resolve()
        if not metadata.is_file():
            raise ValueError(f"artifact sidecar does not exist: {metadata}")
        result["sidecar"] = {"path": str(metadata), "sha256": sha256_file(metadata)}
    return result


def discovered_sidecar(path: str | Path) -> Path | None:
    if not path:
        return None
    source = Path(path)
    if not source.name:
        return None
    candidates = (Path(f"{source}.json"), source.with_suffix(".json"))
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def sidecar_recipe(
    path: str | Path, fallback: str, *, sidecar: str | Path | None = None
) -> str:
    sidecar = Path(sidecar) if sidecar is not None else discovered_sidecar(path)
    if sidecar is None:
        return fallback
    try:
        payload = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read artifact sidecar {sidecar}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"artifact sidecar must contain an object: {sidecar}")
    fields = []
    for key in (
        "model",
        "variant",
        "precision_mode",
        "deform_core",
        "num_queries",
        "eval_idx",
        "cascade",
    ):
        value = payload.get(key)
        if value is not None:
            fields.append(f"{key}={value}")
    return ";".join(fields) or fallback


def package_runtime(label: str, package: str) -> str:
    version = package_version(package) or "unknown"
    return f"{label} {version}"


def gpu_identity() -> list[dict] | None:
    fields = ("index", "name", "uuid", "memory.total", "driver_version")
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if query.returncode != 0:
        return None
    rows = []
    for values in csv.reader(query.stdout.splitlines(), skipinitialspace=True):
        if not values:
            continue
        if len(values) != len(fields):
            return None
        index, name, uuid, memory_mib, driver = (value.strip() for value in values)
        try:
            parsed_index = int(index)
            parsed_memory = int(memory_mib)
        except ValueError:
            return None
        rows.append(
            {
                "index": parsed_index,
                "name": name,
                "uuid": uuid,
                "memory_mib": parsed_memory,
                "driver_version": driver,
            }
        )
    return rows or None


def environment_metadata() -> dict:
    gpus = gpu_identity()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": package_version("numpy"),
        "opencv": package_version("opencv-python-headless") or package_version("opencv-python"),
        "pycocotools": package_version("pycocotools"),
        "torch": package_version("torch"),
        "cuda_runtime_package": package_version("cuda-runtime"),
        "onnxruntime_gpu": package_version("onnxruntime-gpu"),
        "tensorrt": package_version("tensorrt"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_driver_versions": (
            sorted({gpu["driver_version"] for gpu in gpus}) if gpus else None
        ),
        "nvidia_gpus": gpus,
    }


def paths_alias(first: str | Path, second: str | Path) -> bool:
    left = Path(first).resolve()
    right = Path(second).resolve()
    if left == right:
        return True
    if left.exists() and right.exists():
        try:
            return os.path.samefile(left, right)
        except OSError:
            return False
    return False


def atomic_json(
    path: str | Path, payload: dict, *, protected=(), overwrite: bool = False, sort_keys=False
) -> None:
    target = Path(path).resolve()
    for source in protected:
        if source and paths_alias(target, source):
            raise ValueError(f"report output aliases input artifact: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise ValueError(f"output already exists: {target}; pass --overwrite to replace it")
    if target.exists() and not target.is_file():
        raise ValueError(f"output is not a file: {target}")
    encoded = json.dumps(payload, indent=2, allow_nan=False, sort_keys=sort_keys) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary_path, target)
        else:
            try:
                os.link(temporary_path, target)
            except FileExistsError as exc:
                raise ValueError(
                    f"output already exists: {target}; pass --overwrite to replace it"
                ) from exc
            except OSError as exc:
                raise ValueError(f"cannot publish output {target}: {exc}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)
