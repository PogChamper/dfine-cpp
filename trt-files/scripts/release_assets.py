#!/usr/bin/env python3
"""Stage and verify GitHub release assets, replacing the hand-run SHA256SUMS ritual.

``assemble`` validates the 20 model files against the release grammar
  dfine_{n,s,m,l,x}_{op19,slim}.{onnx,json}
before anything is staged: every graph paired with its sidecar (and vice versa),
sidecar parseable, sidecar precision matching the name's recipe suffix (_slim =
fp16, _op19 = fp32 — the same contract the CLI's _check_onnx_precision enforces),
opset 19 on BOTH recipes (the v0.3 surgical fp16 hard-requires an opset-19 base),
and no stray dfine_* files (typo protection). It then copies the models plus the
gated wheel into --out and writes SHA256SUMS over all 21 payload files — the
wheel is hashed too, because the v0.3.1 audit caught it missing from the
hand-assembled manifest.

``verify`` downloads a published release with ``gh release download`` and runs
``sha256sum -c SHA256SUMS`` on it, additionally refusing assets the manifest
does not cover (a failure mode ``sha256sum -c`` alone cannot see).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

SIZES = ("n", "s", "m", "l", "x")
RECIPES = {"op19": "fp32", "slim": "fp16"}  # recipe suffix -> precision its sidecar must carry


def _validate_sidecar(sidecar: Path, precision: str) -> None:
    # Same contract as the CLI's _check_onnx_precision: the sidecar records what
    # the converter produced (no key = legacy fp32 export), and an unparseable
    # sidecar must not silently DISABLE the check.
    try:
        meta = json.loads(sidecar.read_text())
    except (OSError, ValueError) as e:
        raise SystemExit(f"{sidecar.name} cannot be parsed ({e}); re-export before staging")
    actual = meta.get("precision", "fp32")
    if actual != precision:
        raise SystemExit(f"{sidecar.name}: sidecar precision is {actual} but the name's "
                         f"recipe suffix requires {precision}")
    if meta.get("opset") != 19:
        raise SystemExit(f"{sidecar.name}: opset is {meta.get('opset')} but release assets are "
                         "opset 19 (the surgical fp16 recipe hard-requires an opset-19 base)")


def assemble(args: argparse.Namespace) -> None:
    input_dir = Path(args.input).resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"--input is not a directory: {input_dir}")
    wheel = Path(args.wheel).resolve()
    if not wheel.is_file():
        raise SystemExit(f"wheel not found: {wheel}")

    expected = {f"dfine_{s}_{r}.{ext}" for s in SIZES for r in RECIPES for ext in ("onnx", "json")}
    present = {p.name for p in input_dir.glob("dfine_*") if p.is_file()}
    extra = sorted(present - expected)
    if extra:
        raise SystemExit(f"unexpected dfine_* files in {input_dir}: {', '.join(extra)} "
                         "(release grammar is dfine_{n,s,m,l,x}_{op19,slim}.{onnx,json})")
    missing = sorted(expected - present)
    if missing:
        raise SystemExit(f"missing from {input_dir}: {', '.join(missing)} "
                         "(a graph never ships without its sidecar, nor a sidecar alone)")
    for size in SIZES:
        for recipe, precision in RECIPES.items():
            _validate_sidecar(input_dir / f"dfine_{size}_{recipe}.json", precision)

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in sorted(expected):
        shutil.copy2(input_dir / name, out / name)
    shutil.copy2(wheel, out / wheel.name)
    names = sorted(expected | {wheel.name})
    lines = [f"{hashlib.sha256((out / n).read_bytes()).hexdigest()}  {n}" for n in names]
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    print(f"[assemble] staged {len(names)} payload files + SHA256SUMS -> {out}")


def _default_repo() -> str:
    # The same OWNER/NAME gh itself would infer from the cwd's git remote, made
    # explicit so the summary names the repo that was actually verified.
    out = subprocess.run(["gh", "repo", "view", "--json", "nameWithOwner",
                          "--jq", ".nameWithOwner"], capture_output=True, text=True)
    repo = out.stdout.strip()
    if out.returncode != 0 or not repo:
        raise SystemExit("cannot resolve the repo via `gh repo view`; pass --repo OWNER/NAME")
    return repo


def verify(args: argparse.Namespace) -> None:
    repo = args.repo or _default_repo()
    with tempfile.TemporaryDirectory(prefix="dfine-release-verify-") as td:
        print(f"[verify] downloading {repo} {args.tag} -> {td}")
        dl = subprocess.run(["gh", "release", "download", args.tag, "--repo", repo, "--dir", td])
        if dl.returncode != 0:
            raise SystemExit(f"gh release download failed for {repo} {args.tag}")
        sums = Path(td) / "SHA256SUMS"
        if not sums.is_file():
            raise SystemExit(f"release {args.tag} has no SHA256SUMS asset")
        chk = subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=td,
                             capture_output=True, text=True)
        for line in chk.stdout.splitlines():
            print(f"[verify] {line}")
        if chk.stderr.strip():
            print(f"[verify] {chk.stderr.strip()}")
        # `sha256sum -c` only checks files the manifest NAMES — an uploaded asset
        # missing from SUMS (how the v0.3.1 wheel nearly shipped) passes silently,
        # so coverage is checked explicitly.
        covered = {ln.split("  ", 1)[1] for ln in sums.read_text().splitlines() if "  " in ln}
        uncovered = sorted(p.name for p in Path(td).iterdir()
                           if p.name != "SHA256SUMS" and p.name not in covered)
        for name in uncovered:
            print(f"[verify] {name}: NOT IN SHA256SUMS")
        ok = sum(1 for ln in chk.stdout.splitlines() if ln.endswith(": OK"))
        bad = sum(1 for ln in chk.stdout.splitlines() if ": FAILED" in ln)
        print(f"[verify] {ok} OK, {bad} FAILED, {len(uncovered)} not in SHA256SUMS")
        if chk.returncode != 0 or uncovered:
            raise SystemExit(1)
        print(f"[verify] {repo} {args.tag}: all assets match SHA256SUMS")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage (assemble) or check (verify) release assets")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("assemble",
                       help="validate the 20 model files, stage them + the wheel, write SHA256SUMS")
    a.add_argument("--input", required=True,
                   help="directory holding dfine_{n,s,m,l,x}_{op19,slim}.{onnx,json}")
    a.add_argument("--wheel", required=True,
                   help="the gated wheel (staged and hashed into SHA256SUMS with the models)")
    a.add_argument("--out", required=True, help="staging directory for the upload")
    v = sub.add_parser("verify",
                       help="download a published release and run sha256sum -c on it")
    v.add_argument("--tag", required=True, help="release tag, e.g. v0.3.1")
    v.add_argument("--repo", default=None, help="OWNER/NAME (default: `gh repo view` in the cwd)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    {"assemble": assemble, "verify": verify}[args.cmd](args)
