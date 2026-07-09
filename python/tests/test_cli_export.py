"""`dfine export` recipe wiring (mocked subprocess — no torch/TensorRT needed):
--precision fp16 must produce the v0.3 surgical/slim tier on an opset-19 base,
fp16-legacy the v0.2 tier on opset 16, and unvalidated combinations must be
refused. Regression for the v0.3.0 drift where the CLI silently produced the
legacy tier while the README advertised surgical numbers."""

from __future__ import annotations

import pytest

from dfine import cli


@pytest.fixture()
def harness(monkeypatch, tmp_path):
    """Record every subprocess the CLI would launch instead of running it."""
    calls: list[list[str]] = []
    monkeypatch.setattr(cli.subprocess, "run",
                        lambda cmd, check=True: calls.append([str(c) for c in cmd]))
    monkeypatch.setattr(cli, "_seg_dir", lambda: tmp_path)
    monkeypatch.setattr(cli, "_scripts_dir", lambda: tmp_path / "scripts")
    monkeypatch.setattr(cli, "_cache_dir", lambda: tmp_path / "cache")
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"stub")
    return calls, str(ckpt)


def script(call: list[str]) -> str:
    return next(c.rsplit("/", 1)[-1] for c in call if c.endswith(".py"))


def flag(call: list[str], name: str) -> str | None:
    return call[call.index(name) + 1] if name in call else None


def test_fp16_is_surgical_slim_on_opset19(harness):
    calls, ckpt = harness
    assert cli.main(["export", "--model", "m", "--precision", "fp16",
                     "--checkpoint", ckpt]) == 0
    assert [script(c) for c in calls] == ["export_dfine_onnx.py", "convert_fp16_surgical.py"]
    assert flag(calls[0], "--opset") == "19"
    assert "--slim" in calls[1]
    assert flag(calls[1], "--output").endswith("dfine_m_slim.onnx")


def test_fp16_legacy_keeps_the_v02_tier(harness):
    calls, ckpt = harness
    assert cli.main(["export", "--model", "m", "--precision", "fp16-legacy",
                     "--checkpoint", ckpt]) == 0
    assert [script(c) for c in calls] == ["export_dfine_onnx.py", "convert_fp16.py"]
    assert flag(calls[0], "--opset") == "16"
    assert flag(calls[1], "--output").endswith("dfine_m_fp16_st.onnx")


def test_fp32_default_is_opset19(harness):
    calls, ckpt = harness
    assert cli.main(["export", "--model", "m", "--checkpoint", ckpt]) == 0
    assert [script(c) for c in calls] == ["export_dfine_onnx.py"]
    assert flag(calls[0], "--opset") == "19"


def test_unvalidated_opset_pairings_are_refused(harness):
    calls, ckpt = harness
    with pytest.raises(SystemExit, match="fp16-legacy"):
        cli.main(["export", "--precision", "fp16", "--opset", "16", "--checkpoint", ckpt])
    with pytest.raises(SystemExit, match="opset-16"):
        cli.main(["export", "--precision", "fp16-legacy", "--opset", "19", "--checkpoint", ckpt])
    assert calls == []  # refused before any subprocess


def test_custom_model_args_passthrough(harness):
    calls, ckpt = harness
    assert cli.main(["export", "--model", "s", "--checkpoint", ckpt,
                     "--num-classes", "3", "--class-names", "a,b,c",
                     "--allow-partial-checkpoint"]) == 0
    assert flag(calls[0], "--num-classes") == "3"
    assert flag(calls[0], "--class-names") == "a,b,c"
    assert "--allow-partial-checkpoint" in calls[0]


def test_resolver_prefers_the_production_tier():
    assert cli._ONNX_SUFFIXES["fp16"][0] == "_slim"
