"""validation_report.py — the external-validation-matrix tool (docs/VALIDATION.md).
Pure parts only: env collection always returns the full key set with "unknown"
fallbacks, report.json/report.md land together and agree, --check-sums classifies
match/mismatch/not-listed, and the build/bench subprocess plumbing passes the
right flags. Every subprocess is mocked — no GPU, nvidia-smi or tensorrt needed."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

BENCH_STDOUT = """\
dfine_bench: engine=dfine_m_slim.engine  variant=m  input=640x640  src=640x480  warmup=20 iters=200
batch  total_p50   total_p90   total_p99   pre_ms      infer_ms    decode_ms   img/s
1      3.470       3.512       3.601       0.512       2.401       0.301       288.2
8      15.190      15.402      15.822      3.902       9.517       1.204       526.6
peak GPU mem (engine+buffers): 812 MiB / 16376 total
"""


@pytest.fixture()
def mod():
    """A fresh module per test so monkeypatched attributes never leak."""
    spec = importlib.util.spec_from_file_location(
        "validation_report", REPO / "trt-files/scripts/validation_report.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _offline(mod, monkeypatch):
    """No nvidia-smi/git/nvcc, no importable tensorrt/dfine — a GPU-less stranger."""
    monkeypatch.setattr(mod, "_run", lambda *a, **k: None)
    monkeypatch.setattr(mod, "_module_version", lambda name: "unknown")


def _fake_assets(tmp_path, precision="fp16"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    onnx = tmp_path / "dfine_m_slim.onnx"
    onnx.write_bytes(b"not-a-real-graph")
    sidecar = onnx.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "variant": "m", "precision": precision, "opset": 19, "num_classes": 80,
        "precision_mode": "strongly_typed_onnx_fp16" if precision == "fp16" else "fp32"}))
    return onnx, sidecar


def test_env_all_keys_present_with_unknown_fallbacks(mod, monkeypatch):
    _offline(mod, monkeypatch)
    env = mod.collect_env()
    assert set(env) == set(mod.ENV_KEYS)
    for key in ("gpu_name", "compute_cap", "driver", "cuda", "tensorrt", "dfine", "commit"):
        assert env[key] == "unknown"
    for key in ("os", "kernel", "python"):  # platform facts are always determinable
        assert env[key] and env[key] != "unknown"
    assert isinstance(env["wsl"], bool)


def test_reports_written_and_agree(mod, monkeypatch, tmp_path):
    onnx, sidecar = _fake_assets(tmp_path)
    _offline(mod, monkeypatch)  # tensorrt "unknown" -> build skipped -> bench skipped
    sums = tmp_path / "SHA256SUMS"
    sums.write_text(f"{hashlib.sha256(onnx.read_bytes()).hexdigest()}  {onnx.name}\n"
                    f"{'0' * 64}  {sidecar.name}\n")
    out = tmp_path / "out"
    returned = mod.main(["--onnx", str(onnx), "--out", str(out), "--check-sums", str(sums)])

    data = json.loads((out / "report.json").read_text())
    md = (out / "report.md").read_text()
    assert data == returned and data["schema"] == 1
    assert data["build"]["skipped"] is True and "tensorrt" in data["build"]["note"]
    assert data["bench"]["skipped"] is True and "no engine" in data["bench"]["note"]
    assert data["checksums"]["onnx"] == "match"
    assert data["checksums"]["sidecar"] == "mismatch"
    # md is a rendering of the same facts
    assert data["onnx"]["sha256"] == hashlib.sha256(onnx.read_bytes()).hexdigest()
    assert data["onnx"]["sha256"] in md and data["onnx"]["sidecar"]["sha256"] in md
    assert data["onnx"]["sidecar"]["opset"] == 19 and "opset=19" in md
    assert data["build"]["note"] in md and data["bench"]["note"] in md
    assert "onnx match" in md and "sidecar mismatch" in md
    for key in mod.ENV_KEYS:
        if isinstance(data["env"][key], str):
            assert data["env"][key] in md


def test_check_sums_match_mismatch_not_listed(mod, tmp_path):
    a, b = tmp_path / "a.onnx", tmp_path / "a.json"
    a.write_bytes(b"graph")
    b.write_text("{}")
    ha = hashlib.sha256(b"graph").hexdigest()
    sums = tmp_path / "SHA256SUMS"
    sums.write_text(f"{ha}  *./a.onnx\n{'0' * 64}  a.json\nnot a checksum line\n")
    res = mod.verify_sums(sums, {
        "onnx": (str(a), ha.upper()),  # case-insensitive
        "sidecar": (str(b), hashlib.sha256(b"{}").hexdigest()),
        "ghost": (str(tmp_path / "c.bin"), "ff" * 32),
    })
    assert res["onnx"] == "match"
    assert res["sidecar"] == "mismatch"
    assert res["ghost"] == "not-listed"


def test_build_flags_follow_sidecar_precision(mod, monkeypatch, tmp_path):
    def fake_run(cmd, **kw):  # stands in for the build_engine.py subprocess
        engine = Path(cmd[cmd.index("--output") + 1])
        engine.write_bytes(b"engine")
        Path(str(engine) + ".json").write_text(json.dumps(
            {"precision": "fp16", "precision_mode": "strongly_typed_onnx_fp16",
             "trt_version": "10.13.0", "sm_arch": "89"}))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    env = dict.fromkeys(mod.ENV_KEYS, "unknown")
    env["tensorrt"] = "10.13.0"
    out = tmp_path / "out"
    out.mkdir()

    onnx, _ = _fake_assets(tmp_path / "fp16_case", precision="fp16")
    rec = mod.build_engine(mod.describe_onnx(onnx), out, env)
    assert rec["ok"] and "--strongly-typed" in rec["command"]
    assert "--no-tf32" in rec["command"]
    assert rec["command"][rec["command"].index("--max-batch") + 1] == "8"
    assert rec["engine_sidecar"]["precision_mode"] == "strongly_typed_onnx_fp16"

    onnx32, _ = _fake_assets(tmp_path / "fp32_case", precision="fp32")
    rec32 = mod.build_engine(mod.describe_onnx(onnx32), out, env)
    assert rec32["ok"] and "--strongly-typed" not in rec32["command"]


def test_bench_output_parsed_and_embedded(mod, monkeypatch, tmp_path):
    bench = tmp_path / "dfine_bench"
    bench.write_text("")  # existence gates the run; execution is mocked below
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=0, stdout=BENCH_STDOUT, stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    rec = mod.run_bench(str(tmp_path / "e.engine"), bench=bench)
    assert rec["ok"] and not rec["skipped"]
    assert [r["batch"] for r in rec["results"]] == [1, 8]
    assert rec["results"][0]["img_per_s"] == 288.2
    assert rec["results"][1]["img_per_s"] == 526.6
    assert calls[0][calls[0].index("--batches") + 1] == "1,8"
    # no binary -> skipped with the documented note, no subprocess call
    rec2 = mod.run_bench(str(tmp_path / "e.engine"), bench=tmp_path / "missing")
    assert rec2["skipped"] and "no dfine_bench binary" in rec2["note"] and len(calls) == 1
