"""`dfine` — a zero-setup command line for the D-FINE-cpp TensorRT runtime.

    dfine predict --model m --image dog.jpg          # detect + print (+ --out to draw)
    dfine info    --model m                           # engine introspection
    dfine build   --model m --precision fp16          # ONNX -> .engine (into the cache)
    dfine export  --model m                            # .pt  -> ONNX  (needs D-FINE-seg)
    dfine bench   --model m --batches 1,2,4,8          # latency/throughput

Engines are resolved from (in order): --engine, the on-disk cache
(~/.cache/dfine), the dev-tree trt-files/engines, or built on demand from an
ONNX. Engines are GPU-arch- and TensorRT-version-specific, so the cache key
includes both.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

MODELS = ("n", "s", "m", "l", "x")
_ENGINE_SUFFIX = {"fp32": "fp32", "fp16": "fp16_st"}
_ONNX_SUFFIX = {"fp32": "", "fp16": "_fp16_st"}
# Known D-FINE-seg checkpoints (relative to the seg source dir).
_CHECKPOINTS = {
    "n": "dfine_n_coco.pt",
    "s": "dfine_s_obj2coco.pt",
    "m": "pretrained/dfine_m_obj2coco.pt",
    "l": "dfine_l_obj2coco.pt",
    "x": "dfine_x_obj2coco.pt",
}


# --------------------------------------------------------------------------- #
# Environment probing
# --------------------------------------------------------------------------- #


def _cache_dir() -> Path:
    d = Path(os.environ.get("DFINE_CACHE", Path.home() / ".cache" / "dfine"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _repo_root() -> Optional[Path]:
    """The dev-tree repo root if we're running from a source checkout, else None."""
    root = Path(__file__).resolve().parents[2]
    return root if (root / "trt-files" / "scripts" / "build_engine.py").exists() else None


def _seg_dir() -> Optional[Path]:
    env = os.environ.get("DFINE_SEG_DIR")
    if env and Path(env).exists():
        return Path(env)
    repo = _repo_root()
    if repo:
        sib = repo.parent / "D-FINE-seg"
        if sib.exists():
            return sib
    return None


def _gpu_arch() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        cap = out.stdout.strip().splitlines()[0].strip()
        return cap.replace(".", "") or "unknown"
    except Exception:
        return "unknown"


def _trt_version() -> str:
    try:
        import tensorrt  # type: ignore

        return tensorrt.__version__
    except Exception:
        return "unknown"


def _have_tensorrt() -> bool:
    try:
        import tensorrt  # noqa: F401

        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Engine / ONNX resolution
# --------------------------------------------------------------------------- #


def _cache_engine_path(model: str, precision: str) -> Path:
    return _cache_dir() / f"dfine_{model}_{precision}-sm{_gpu_arch()}-trt{_trt_version()}.engine"


def _find_onnx(model: str, precision: str, onnx_arg: Optional[str]) -> Optional[Path]:
    if onnx_arg:
        p = Path(onnx_arg)
        return p if p.exists() else None
    name = f"dfine_{model}{_ONNX_SUFFIX[precision]}.onnx"
    for base in (_cache_dir(), (_repo_root() / "trt-files" / "onnx") if _repo_root() else None):
        if base and (base / name).exists():
            return base / name
    return None


def _resolve_engine(
    model: Optional[str],
    engine_arg: Optional[str],
    precision: str,
    onnx_arg: Optional[str],
    allow_build: bool,
) -> Path:
    if engine_arg:
        p = Path(engine_arg)
        if not p.exists():
            raise SystemExit(f"engine not found: {p}")
        return p
    if not model:
        raise SystemExit("specify --engine PATH or --model {n,s,m,l,x}")
    if model not in MODELS:
        raise SystemExit(f"unknown model '{model}' (choose from {', '.join(MODELS)})")

    # 1) cache
    cached = _cache_engine_path(model, precision)
    if cached.exists():
        return cached
    # 2) dev tree
    repo = _repo_root()
    if repo:
        dev = repo / "trt-files" / "engines" / f"dfine_{model}_{_ENGINE_SUFFIX[precision]}.engine"
        if dev.exists():
            return dev
    # 3) build on demand
    if allow_build:
        onnx = _find_onnx(model, precision, onnx_arg)
        if onnx is None:
            raise SystemExit(
                f"no engine or ONNX for model '{model}' ({precision}). Provide --onnx, or run "
                "`dfine export` first (needs the D-FINE-seg source), or pass --engine."
            )
        print(f"[dfine] no cached engine — building {model} ({precision}) from {onnx.name} ...")
        return _build_engine(onnx, cached, precision, max_batch=8)
    raise SystemExit(f"no engine found for model '{model}' ({precision}); pass --engine or build one")


# --------------------------------------------------------------------------- #
# Shelling out to the build/export scripts
# --------------------------------------------------------------------------- #


def _scripts_dir() -> Path:
    repo = _repo_root()
    if not repo:
        raise SystemExit("build/export need the dev-tree scripts (trt-files/scripts); not found")
    return repo / "trt-files" / "scripts"


def _build_engine(onnx: Path, output: Path, precision: str, max_batch: int) -> Path:
    if not _have_tensorrt():
        raise SystemExit("building an engine needs TensorRT — `pip install tensorrt==10.13.*`")
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_scripts_dir() / "build_engine.py"),
        "--onnx", str(onnx),
        "--output", str(output),
        "--max-batch", str(max_batch),
        "--no-tf32",
    ]
    if precision == "fp16":
        cmd += ["--strongly-typed"]  # ONNX is already FP16-typed (convert_fp16.py output)
    print("[dfine] $", " ".join(cmd))
    subprocess.run(cmd, check=True)
    if not output.exists():
        raise SystemExit(f"build reported success but {output} is missing")
    return output


def _convert_fp16(fp32_onnx: Path, output: Path) -> Path:
    """Strongly-typed FP16 ONNX (backbone+encoder FP16, decoder FP32) via convert_fp16.py."""
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_scripts_dir() / "convert_fp16.py"),
        "--onnx", str(fp32_onnx),
        "--output", str(output),
    ]
    print("[dfine] $", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return output


def _export_onnx(model: str, checkpoint: Optional[str], output: Path) -> Path:
    seg = _seg_dir()
    if seg is None:
        raise SystemExit(
            "export needs the D-FINE-seg source (set DFINE_SEG_DIR or place it beside this repo)"
        )
    ckpt = Path(checkpoint) if checkpoint else seg / _CHECKPOINTS[model]
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt} (pass --checkpoint)")
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_scripts_dir() / "export_dfine_onnx.py"),
        "--model-name", model,
        "--checkpoint", str(ckpt),
        "--dfine-src", str(seg),
        "--output", str(output),
    ]
    print("[dfine] $", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return output


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #

_PALETTE = [
    (255, 56, 56), (255, 159, 56), (255, 214, 56), (162, 255, 56), (56, 255, 128),
    (56, 236, 255), (56, 128, 255), (128, 56, 255), (222, 56, 255), (255, 56, 152),
]


def _draw(image_path: str, dets, out_path: str) -> None:
    from PIL import Image, ImageDraw

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for d in dets:
        color = _PALETTE[d.class_id % len(_PALETTE)]
        x1, y1, x2, y2 = d.box.as_tuple()
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{d.class_name} {d.score:.2f}"
        ty = max(0.0, y1 - 12)
        draw.rectangle([x1, ty, x1 + 7 * len(label), ty + 12], fill=color)
        draw.text((x1 + 2, ty), label, fill=(0, 0, 0))
    img.save(out_path)


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #


def cmd_predict(args) -> int:
    import numpy as np
    from PIL import Image

    from .detector import Detector

    engine = _resolve_engine(args.model, args.engine, args.precision, args.onnx, allow_build=True)
    arr = np.asarray(Image.open(args.image).convert("RGB"))
    with Detector(str(engine), threshold=args.threshold) as det:
        dets = det.detect(arr, threshold=args.threshold)
    dets.sort(key=lambda d: d.score, reverse=True)

    if args.json:
        print(json.dumps([d.as_dict() for d in dets], indent=2))
    else:
        print(f"engine: {engine.name}")
        print(f"{len(dets)} detection(s) (thr={args.threshold:.2f})")
        for d in dets:
            x1, y1, x2, y2 = d.box.as_tuple()
            print(f"  {d.class_name:16s} {d.score:.3f}  [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

    if args.out:
        _draw(args.image, dets, args.out)
        print(f"wrote {args.out}")
    return 0


def cmd_info(args) -> int:
    from .detector import Detector

    engine = _resolve_engine(args.model, args.engine, args.precision, args.onnx, allow_build=False)
    with Detector(str(engine)) as det:
        print(f"engine      : {engine}")
        print(f"variant     : {det.variant}")
        print(f"input       : {det.input_width}x{det.input_height}")
        print(f"num_queries : {det.num_queries}")
        print(f"num_classes : {det.num_classes}")
        print(f"max_batch   : {det.max_batch}")
    return 0


def cmd_build(args) -> int:
    onnx = _find_onnx(args.model, args.precision, args.onnx)
    if onnx is None:
        raise SystemExit(
            f"no ONNX for model '{args.model}' ({args.precision}); pass --onnx or run `dfine export`"
        )
    out = Path(args.output) if args.output else _cache_engine_path(args.model, args.precision)
    _build_engine(onnx, out, args.precision, args.max_batch)
    print(f"built {out}")
    return 0


def cmd_export(args) -> int:
    model = args.model
    # Always produce the FP32 ONNX first (its name has no precision suffix, matching
    # _find_onnx for fp32). For fp16, convert it to the strongly-typed FP16 ONNX under
    # the exact name `dfine build --precision fp16` / `predict` look up.
    fp32_out = (
        Path(args.output)
        if (args.output and args.precision == "fp32")
        else _cache_dir() / f"dfine_{model}.onnx"
    )
    _export_onnx(model, args.checkpoint, fp32_out)
    if args.precision == "fp32":
        print(f"exported {fp32_out}")
        return 0
    fp16_out = Path(args.output) if args.output else _cache_dir() / f"dfine_{model}_fp16_st.onnx"
    _convert_fp16(fp32_out, fp16_out)
    print(f"exported {fp16_out}")
    return 0


def cmd_bench(args) -> int:
    engine = _resolve_engine(args.model, args.engine, args.precision, args.onnx, allow_build=False)
    repo = _repo_root()
    bench_bin = repo / "build" / "dfine_bench" if repo else None
    if bench_bin and bench_bin.exists():
        cmd = [str(bench_bin), "--engine", str(engine), "--batches", args.batches]
        print("[dfine] $", " ".join(cmd))
        return subprocess.run(cmd).returncode
    raise SystemExit("dfine_bench binary not found (build it with ./build.sh) — no bench backend")


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def _add_common(sp, *, engine=True, precision=True):
    sp.add_argument("--model", choices=MODELS, help="model size (resolves a cached/dev engine)")
    if engine:
        sp.add_argument("--engine", help="explicit .engine path (overrides --model)")
    if precision:
        sp.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    sp.add_argument("--onnx", help="explicit ONNX path (for build / on-demand build)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dfine", description="D-FINE-cpp TensorRT CLI")
    from . import __version__

    p.add_argument("--version", action="version", version=f"dfine {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("predict", help="detect objects in an image")
    _add_common(pr)
    pr.add_argument("--image", required=True)
    pr.add_argument("--threshold", type=float, default=0.5)
    pr.add_argument("--out", help="write an annotated image here (needs pillow)")
    pr.add_argument("--json", action="store_true", help="print detections as JSON")
    pr.set_defaults(func=cmd_predict)

    pi = sub.add_parser("info", help="print engine introspection")
    _add_common(pi)
    pi.set_defaults(func=cmd_info)

    pb = sub.add_parser("build", help="build a .engine from an ONNX (into the cache)")
    _add_common(pb, engine=False)
    pb.add_argument("--output", help="engine output path (default: cache)")
    pb.add_argument("--max-batch", type=int, default=8)
    pb.set_defaults(func=cmd_build)

    pe = sub.add_parser("export", help="export a checkpoint to ONNX (needs D-FINE-seg)")
    pe.add_argument("--model", choices=MODELS, default="m")
    pe.add_argument("--precision", choices=("fp32", "fp16"), default="fp32",
                    help="fp16 additionally runs convert_fp16 (strongly-typed FP16 ONNX)")
    pe.add_argument("--checkpoint", help="path to a D-FINE .pt (defaults to the known seg ckpt)")
    pe.add_argument("--output", help="ONNX output path (default: cache)")
    pe.set_defaults(func=cmd_export)

    pbe = sub.add_parser("bench", help="benchmark latency/throughput (C++ dfine_bench)")
    _add_common(pbe)
    pbe.add_argument("--batches", default="1,2,4,8")
    pbe.set_defaults(func=cmd_bench)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except subprocess.CalledProcessError as e:
        print(f"[dfine] subprocess failed (exit {e.returncode})", file=sys.stderr)
        return e.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
