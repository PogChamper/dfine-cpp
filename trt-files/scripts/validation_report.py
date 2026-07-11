#!/usr/bin/env python3
"""Produce a comparable report for the cross-GPU validation matrix.

Engines are compiled on the user's GPU from the released ONNX so each submitted
row records the same artifact, recipe, environment, and benchmark contract:

    python trt-files/scripts/validation_report.py --onnx dfine_m_slim.onnx \\
        --check-sums SHA256SUMS --out validation

It records the environment, hashes the ONNX + sidecar, builds an engine via
build_engine.py (the README quickstart recipe), benches it if build/dfine_bench
exists, and writes report.json (schema 1) + report.md to --out. Everything is
best-effort: no nvidia-smi / tensorrt / bench binary degrades to "unknown" or
a "skipped" note, never a crash — a GPU-less machine still gets a useful report.
Stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "build" / "dfine_bench"

# Every report carries exactly these environment keys; unknowable facts are "unknown".
ENV_KEYS = (
    "gpu_name",
    "compute_cap",
    "driver",
    "cuda",
    "tensorrt",
    "os",
    "kernel",
    "wsl",
    "python",
    "dfine",
    "commit",
)


def _run(cmd: list[str], timeout: int = 15) -> str | None:
    """stdout of a successful command, else None (missing binary, timeout, rc!=0)."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout if out.returncode == 0 else None
    except Exception:
        return None


def _module_version(name: str) -> str:
    try:
        return str(getattr(importlib.import_module(name), "__version__", "unknown"))
    except Exception:
        return "unknown"


def _checkout_dfine_version() -> str:
    try:
        text = (REPO / "python/dfine/__init__.py").read_text()
    except OSError:
        return "unknown"
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    return match.group(1) if match else "unknown"


def collect_env() -> dict:
    env = dict.fromkeys(ENV_KEYS, "unknown")
    q = _run(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"])
    if q and q.strip():
        parts = [p.strip() for p in q.strip().splitlines()[0].split(",")]
        if len(parts) == 3:
            env["gpu_name"], env["compute_cap"], env["driver"] = (p or "unknown" for p in parts)
    # "CUDA Version" in the nvidia-smi banner is the driver's supported runtime;
    # fall back to a toolkit nvcc if there is no driver at all.
    m = re.search(r"CUDA Version:\s*([\d.]+)", _run(["nvidia-smi"]) or "") or re.search(
        r"release\s+([\d.]+)", _run(["nvcc", "--version"]) or ""
    )
    if m:
        env["cuda"] = m.group(1)
    env["tensorrt"] = _module_version("tensorrt")
    env["dfine"] = _checkout_dfine_version()
    uname = platform.uname()
    env["os"], env["kernel"] = uname.system or "unknown", uname.release or "unknown"
    try:
        env["wsl"] = "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        env["wsl"] = False
    env["python"] = platform.python_version()
    commit = _run(
        ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"]
    )
    if commit and commit.strip():
        env["commit"] = commit.strip()
    return env


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe_onnx(onnx_path: Path) -> dict:
    """Hashes + contract key fields of a release ONNX and its .json sidecar."""
    onnx_path = Path(onnx_path)
    info = {
        "path": str(onnx_path),
        "sha256": _sha256(onnx_path),
        "size_bytes": onnx_path.stat().st_size,
        "sidecar": None,
    }
    sc = onnx_path.with_suffix(".json")
    if sc.is_file():
        meta = json.loads(sc.read_text())
        info["sidecar"] = {"path": str(sc), "sha256": _sha256(sc)}
        info["sidecar"].update(
            {
                k: meta.get(k, "unknown")
                for k in ("variant", "precision", "precision_mode", "opset", "num_classes")
            }
        )
    return info


def verify_sums(sums_path: Path, files: dict[str, tuple[str, str]]) -> dict:
    """files = {label: (path, sha256)} -> {label: match|mismatch|not-listed|duplicate}.
    Lines are standard `sha256sum` output (`<hex>  [*]name`); matched by basename."""
    table = {}
    duplicates = set()
    for line in Path(sums_path).read_text().splitlines():
        m = re.match(r"^([0-9a-fA-F]{64})[ \t]+\*?(.+)$", line.strip())
        if m:
            name = Path(m.group(2).strip()).name
            if name in table:
                duplicates.add(name)
            table[name] = m.group(1).lower()
    result = {"file": str(sums_path)}
    for label, (path, digest) in files.items():
        name = Path(path).name
        expected = table.get(name)
        result[label] = (
            "duplicate"
            if name in duplicates
            else "not-listed"
            if expected is None
            else "match"
            if expected == digest.lower()
            else "mismatch"
        )
    return result


def checksum_failures(checks: dict | None) -> dict[str, str]:
    if checks is None:
        return {}
    return {key: value for key, value in checks.items() if key != "file" and value != "match"}


def build_engine(onnx_info: dict, out_dir: Path, env: dict) -> dict:
    """Compile the release ONNX with the README quickstart recipe (build_engine.py
    --no-tf32 --max-batch 8, plus --strongly-typed when the sidecar says the ONNX
    itself is fp16-typed). Skipped, not failed, when tensorrt is not importable."""
    rec = {
        "skipped": False,
        "ok": False,
        "wall_s": None,
        "command": None,
        "engine": None,
        "engine_sidecar": None,
        "note": None,
    }
    if env.get("tensorrt", "unknown") == "unknown":
        rec.update(skipped=True, note="build skipped (tensorrt not importable)")
        return rec
    onnx_path = Path(onnx_info["path"])
    engine = Path(out_dir) / (onnx_path.stem + ".engine")
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "build_engine.py"),
        "--onnx",
        str(onnx_path),
        "--output",
        str(engine),
        "--no-tf32",
        "--max-batch",
        "8",
    ]
    if (onnx_info.get("sidecar") or {}).get("precision") == "fp16":
        cmd.append("--strongly-typed")  # fp16-typed graph: precision comes from ONNX types
    rec["command"] = cmd
    print(f"[validate] building engine (can take minutes): {' '.join(cmd[1:])}")
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        rec["ok"] = proc.returncode == 0
        tail = (proc.stdout + proc.stderr).splitlines()[-20:]
    except Exception as e:
        tail = [f"{type(e).__name__}: {e}"]
    rec["wall_s"] = round(time.monotonic() - t0, 1)
    if rec["ok"]:
        rec["engine"] = str(engine)
        # Appended name first — build_engine.py uses it when engine and ONNX share a stem.
        for p in (Path(str(engine) + ".json"), engine.with_suffix(".json")):
            if p.is_file():
                rec["engine_sidecar"] = json.loads(p.read_text())
                break
    else:
        rec["note"] = "build FAILED; last output lines:\n" + "\n".join(tail)
    return rec


# A dfine_bench result row: batch, p50/p90/p99, pre, infer, decode, img/s.
_BENCH_ROW = re.compile(r"^\s*(\d+)" + r"\s+([\d.]+)" * 7 + r"\s*$")


def parse_bench_stdout(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        m = _BENCH_ROW.match(line)
        if m:
            rows.append(
                {
                    "batch": int(m.group(1)),
                    "total_p50_ms": float(m.group(2)),
                    "img_per_s": float(m.group(8)),
                }
            )
    return rows


def run_bench(engine: str | None, bench: Path | None = None) -> dict:
    rec = {"skipped": False, "ok": False, "command": None, "results": [], "note": None}
    bench = Path(bench) if bench is not None else BENCH
    if engine is None:
        rec.update(skipped=True, note="bench skipped (no engine built)")
        return rec
    if not bench.is_file():
        rec.update(skipped=True, note="bench skipped (no dfine_bench binary)")
        return rec
    cmd = [str(bench), "--engine", str(engine), "--batches", "1,8"]
    rec["command"] = cmd
    print(f"[validate] benching: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        rec["results"] = parse_bench_stdout(proc.stdout)
        batches = sorted(row["batch"] for row in rec["results"])
        rec["ok"] = proc.returncode == 0 and batches == [1, 8]
        if proc.returncode != 0:
            err = proc.stderr.strip().splitlines() or ["(no stderr)"]
            rec["note"] = "bench FAILED: " + err[-1]
        elif not rec["ok"]:
            rec["note"] = f"bench FAILED: expected batches 1 and 8, parsed {batches}"
    except Exception as e:
        rec["note"] = f"bench FAILED: {type(e).__name__}: {e}"
    return rec


def render_md(report: dict) -> str:
    env, onnx = report["env"], report["onnx"]
    os_line = f"{env['os']} {env['kernel']}" + (" (WSL2)" if env["wsl"] is True else "")
    lines = [
        "# D-FINE validation report",
        "",
        f"Generated {report['generated_utc']} — schema {report['schema']}",
        "",
        "## Environment",
        "",
        "| fact | value |",
        "|---|---|",
    ]
    lines += [
        f"| {k} | {v} |"
        for k, v in (
            ("GPU", env["gpu_name"]),
            ("Compute cap (SM)", env["compute_cap"]),
            ("Driver", env["driver"]),
            ("CUDA", env["cuda"]),
            ("TensorRT", env["tensorrt"]),
            ("OS / kernel", os_line),
            ("Python", env["python"]),
        ("dfine tooling", env["dfine"]),
            ("repo commit", env["commit"]),
        )
    ]
    lines += [
        "",
        "## Artifact",
        "",
        "| file | sha256 | key fields |",
        "|---|---|---|",
        f"| {Path(onnx['path']).name} | `{onnx['sha256']}` | |",
    ]
    sc = onnx["sidecar"]
    if sc:
        lines.append(
            f"| {Path(sc['path']).name} | `{sc['sha256']}` | "
            f"precision={sc['precision']} opset={sc['opset']} "
            f"num_classes={sc['num_classes']} |"
        )
    else:
        lines.append("| (sidecar) | MISSING | |")
    sums = report["checksums"]
    if sums:
        verdicts = ", ".join(f"{k} {v}" for k, v in sums.items() if k != "file")
        lines += ["", f"SHA256SUMS ({Path(sums['file']).name}): {verdicts}"]
    build = report["build"]
    lines += ["", "## Engine build", ""]
    if build["skipped"]:
        lines.append(build["note"])
    elif build["ok"]:
        es = build["engine_sidecar"] or {}
        lines.append(
            f"OK in {build['wall_s']} s — precision={es.get('precision', '?')} "
            f"mode={es.get('precision_mode', '?')} trt={es.get('trt_version', '?')} "
            f"sm={es.get('sm_arch', '?')} batch {es.get('min_batch', '?')}/"
            f"{es.get('opt_batch', '?')}/{es.get('max_batch', '?')}"
        )
        cmd = [Path(c).name if i < 2 else c for i, c in enumerate(build["command"])]
        lines.append(f"(`{' '.join(cmd)}`)")
    else:
        lines += [f"FAILED after {build['wall_s']} s", "", "```", build["note"] or "", "```"]
    bench = report["bench"]
    lines += ["", "## Bench (dfine_bench, batches 1,8)", ""]
    if bench["skipped"] or not bench["ok"]:
        lines.append(bench["note"] or "bench ran but produced no result rows")
    else:
        lines += ["| batch | total p50 (ms) | img/s |", "|---|---|---|"]
        lines += [
            f"| {r['batch']} | {r['total_p50_ms']} | {r['img_per_s']} |" for r in bench["results"]
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        raise SystemExit(f"--onnx {onnx_path}: no such file")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = collect_env()
    onnx_info = describe_onnx(onnx_path)
    checks = None
    if args.check_sums:
        files = {"onnx": (onnx_info["path"], onnx_info["sha256"])}
        if onnx_info["sidecar"]:
            files["sidecar"] = (onnx_info["sidecar"]["path"], onnx_info["sidecar"]["sha256"])
        checks = verify_sums(args.check_sums, files)
        if not onnx_info["sidecar"]:
            checks["sidecar"] = "missing-local"
        failures = checksum_failures(checks)
        if failures:
            detail = ", ".join(f"{name}={status}" for name, status in sorted(failures.items()))
            raise SystemExit(f"checksum verification failed: {detail}")
    build = build_engine(onnx_info, out_dir, env)
    bench = run_bench(build["engine"])
    report = {
        "schema": 1,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "env": env,
        "onnx": onnx_info,
        "checksums": checks,
        "build": build,
        "bench": bench,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out_dir / "report.md").write_text(render_md(report))
    print(
        f"[validate] wrote {out_dir / 'report.json'} + report.md — "
        "submit both per docs/VALIDATION.md"
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build + bench a released ONNX and write a submittable validation report"
    )
    p.add_argument(
        "--onnx",
        required=True,
        help="downloaded release ONNX (its .json sidecar is expected next to it)",
    )
    p.add_argument("--out", required=True, help="directory for report.json + report.md")
    p.add_argument(
        "--check-sums",
        default=None,
        help="release SHA256SUMS file to verify the ONNX + sidecar against",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    main()
