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
# ONNX stem suffixes tried per precision, in preference order. `_slim` first: it is
# both the v0.3 release surgical FP16 asset name AND what `dfine export --precision
# fp16` writes since v0.3.1, so the production tier always wins. `_fp16_st` (the
# legacy decoder-FP32 tier, still produced by --precision fp16-legacy) is found when
# it is the only candidate; _find_onnx warns whenever several names match.
_ONNX_SUFFIXES = {"fp32": ("", "_op19"), "fp16": ("_slim", "_fp16_st")}
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
        if not p.exists():
            raise SystemExit(f"ONNX not found: {p}")
        return p
    # Location-major (the cache holds user-produced exports and must keep shadowing the
    # dev tree, as in v0.2); within a location, prefer the newer release naming.
    candidates = []
    for base in (_cache_dir(), (_repo_root() / "trt-files" / "onnx") if _repo_root() else None):
        for suffix in _ONNX_SUFFIXES[precision]:
            name = f"dfine_{model}{suffix}.onnx"
            if base and (base / name).exists():
                candidates.append(base / name)
    if len(candidates) > 1:
        others = ", ".join(str(c) for c in candidates[1:])
        print(f"[dfine] using {candidates[0]} (also found: {others} — pass --onnx to override)")
    return candidates[0] if candidates else None


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


def _build_engine_script() -> Path:
    """build_engine.py from the dev tree, or the copy the wheel bundles.

    The script is self-contained (tensorrt + stdlib only), so bundling a snapshot
    lets a wheel-only install go release-ONNX -> engine without a repo checkout.
    """
    if _repo_root():
        return _scripts_dir() / "build_engine.py"
    bundled = Path(__file__).resolve().parent / "_scripts" / "build_engine.py"
    if bundled.exists():
        return bundled
    raise SystemExit(
        "build_engine.py not found — this install has neither the dev tree "
        "(trt-files/scripts) nor the wheel-bundled copy (dfine/_scripts)"
    )


def _check_onnx_precision(onnx: Path, precision: str) -> None:
    """Refuse a silent precision mismatch: a strongly-typed build follows the ONNX
    tensor types, so requesting fp16 on an FP32-typed export (or vice versa) would
    quietly produce an engine of the other precision. The sidecar records what the
    converter produced; without one we cannot tell, so no check."""
    sidecar = onnx.with_suffix(".json")
    if not sidecar.exists():
        return
    try:
        actual = json.loads(sidecar.read_text()).get("precision", "fp32")
    except (OSError, ValueError) as e:
        # An unreadable sidecar must not silently DISABLE this safety check.
        raise SystemExit(f"{sidecar.name} exists but cannot be parsed ({e}); "
                         "fix or remove it before building")
    if actual != precision:
        raise SystemExit(
            f"{onnx.name} is an {actual} export (per its sidecar) but --precision is "
            f"{precision}; pass --precision {actual} or pick the matching release ONNX "
            f"({'dfine_<size>_slim' if precision == 'fp16' else 'dfine_<size>_op19'})"
        )


def _build_engine(onnx: Path, output: Path, precision: str, max_batch: int,
                  opt_batch: int = 1) -> Path:
    if not _have_tensorrt():
        raise SystemExit("building an engine needs TensorRT — `pip install tensorrt==10.13.*`")
    _check_onnx_precision(onnx, precision)
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_build_engine_script()),
        "--onnx", str(onnx),
        "--output", str(output),
        "--max-batch", str(max_batch),
        "--opt-batch", str(opt_batch),
        "--no-tf32",
    ]
    if precision == "fp16":
        # ONNX is already FP16-typed (convert_fp16.py / convert_fp16_surgical.py output)
        cmd += ["--strongly-typed"]
    print("[dfine] $", " ".join(cmd))
    subprocess.run(cmd, check=True)
    if not output.exists():
        raise SystemExit(f"build reported success but {output} is missing")
    return output


def _convert_fp16(fp32_onnx: Path, output: Path) -> Path:
    """Legacy strongly-typed FP16 (backbone+encoder FP16, whole decoder FP32) via
    convert_fp16.py — the v0.2 tier, kept as --precision fp16-legacy."""
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


def _convert_fp16_surgical(fp32_onnx: Path, output: Path) -> Path:
    """Production FP16 (v0.3 surgical/slim: FP16 decoder with the FDR/deform FP32
    island) via convert_fp16_surgical.py. Needs an opset >= 19 base graph."""
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_scripts_dir() / "convert_fp16_surgical.py"),
        "--onnx", str(fp32_onnx),
        "--output", str(output),
        "--slim",
    ]
    print("[dfine] $", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return output


def _export_onnx(model: str, checkpoint: Optional[str], output: Path,
                 opset: Optional[int] = None, num_classes: Optional[int] = None,
                 class_names: Optional[str] = None, allow_partial: bool = False) -> Path:
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
    if opset is not None:
        cmd += ["--opset", str(opset)]
    if num_classes is not None:
        cmd += ["--num-classes", str(num_classes)]
    if class_names:
        cmd += ["--class-names", class_names]
    if allow_partial:
        cmd += ["--allow-partial-checkpoint"]
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
    if not args.model and not args.onnx:
        raise SystemExit("specify --model {n,s,m,l,x} or --onnx PATH")
    if not args.model and not args.output:
        # Without a model there is no cache key the other subcommands would ever
        # resolve, so a cache-named engine would be a dead end.
        raise SystemExit("with --onnx alone, pass --output PATH "
                         "(or add --model to cache under a preset name)")
    if args.opt_batch > args.max_batch:
        raise SystemExit(f"--opt-batch {args.opt_batch} exceeds --max-batch {args.max_batch}")
    onnx = _find_onnx(args.model, args.precision, args.onnx)
    if onnx is None:
        raise SystemExit(
            f"no ONNX for model '{args.model}' ({args.precision}); pass --onnx or run `dfine export`"
        )
    out = Path(args.output) if args.output else _cache_engine_path(args.model, args.precision)
    _build_engine(onnx, out, args.precision, args.max_batch, args.opt_batch)
    print(f"built {out}")
    return 0


def cmd_export(args) -> int:
    model = args.model
    # Validated opset/recipe pairings only. fp16 = the v0.3 production surgical/slim
    # recipe, which hard-requires an opset-19 base (opset-16 decomposed LayerNorm
    # miscompiles under fine-grained FP16 in TRT 10.13); fp16-legacy = the measured
    # v0.2 tier on its original opset-16 base. Unmeasured combinations are refused
    # rather than silently exported.
    if args.precision == "fp16" and args.opset is not None and args.opset < 19:
        raise SystemExit(f"--precision fp16 (surgical) needs --opset >= 19, got {args.opset}; "
                         "use --precision fp16-legacy for the opset-16 tier")
    if args.precision == "fp16-legacy" and args.opset not in (None, 16):
        raise SystemExit("--precision fp16-legacy is the validated opset-16 v0.2 tier; "
                         "drop --opset or use --precision fp16")
    opset = args.opset if args.opset is not None else (16 if args.precision == "fp16-legacy"
                                                       else 19)

    # Always produce the FP32 ONNX first (its name has no precision suffix, matching
    # _find_onnx for fp32). For the fp16 recipes, convert it under the name
    # `dfine build --precision fp16` / `predict` resolve first (_ONNX_SUFFIXES).
    fp32_out = (
        Path(args.output)
        if (args.output and args.precision == "fp32")
        else _cache_dir() / f"dfine_{model}.onnx"
    )
    _export_onnx(model, args.checkpoint, fp32_out, opset,
                 num_classes=args.num_classes, class_names=args.class_names,
                 allow_partial=args.allow_partial_checkpoint)
    if args.precision == "fp32":
        print(f"exported {fp32_out}")
        return 0
    if args.precision == "fp16":
        out = Path(args.output) if args.output else _cache_dir() / f"dfine_{model}_slim.onnx"
        _convert_fp16_surgical(fp32_out, out)
    else:  # fp16-legacy
        out = Path(args.output) if args.output else _cache_dir() / f"dfine_{model}_fp16_st.onnx"
        _convert_fp16(fp32_out, out)
    print(f"exported {out}")
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
    pb.add_argument("--opt-batch", type=int, default=1,
                    help="batch size TRT optimizes tactics for (8 for batch serving; "
                         "1, the default, for lowest single-image latency)")
    pb.set_defaults(func=cmd_build)

    pe = sub.add_parser("export", help="export a checkpoint to ONNX (needs D-FINE-seg)")
    pe.add_argument("--model", choices=MODELS, default="m")
    pe.add_argument("--precision", choices=("fp32", "fp16", "fp16-legacy"), default="fp32",
                    help="fp16 = the v0.3 production surgical/slim recipe (opset 19); "
                         "fp16-legacy = the v0.2 decoder-FP32 tier (opset 16)")
    pe.add_argument("--checkpoint", help="path to a D-FINE .pt (defaults to the known seg ckpt)")
    pe.add_argument("--num-classes", type=int, default=None,
                    help="class count of the checkpoint (default 80); a mismatch aborts "
                         "the export instead of silently dropping the classifier head")
    pe.add_argument("--class-names", default=None,
                    help="display names for the sidecar: a file (one per line) or a comma list")
    pe.add_argument("--allow-partial-checkpoint", action="store_true",
                    help="research only: keep exporting when checkpoint tensors are "
                         "missing/mismatched (the sidecar records the partial load)")
    pe.add_argument("--output", help="ONNX output path (default: cache)")
    pe.add_argument("--opset", type=int, default=None,
                    help="ONNX opset (default: 19; fp16-legacy uses its validated 16)")
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
