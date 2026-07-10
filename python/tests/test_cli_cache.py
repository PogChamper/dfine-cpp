"""Engine-cache identity: an engine is bound to the exact ONNX (+sidecar +batch
profile) it was built from, an explicit --onnx can never lose to a cache entry
built from something else, and provenance-less fallbacks are never picked
silently among several candidates. Regression for the v0.3.0 shadowing bug
(a stale COCO engine silently served a fresh custom export)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dfine import cli


@pytest.fixture()
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(cli, "_repo_root", lambda: None)   # no dev-tree fallback
    monkeypatch.setattr(cli, "_gpu_arch", lambda: "89")
    monkeypatch.setattr(cli, "_trt_version", lambda: "10.13")
    builds: list[tuple[Path, Path]] = []

    def fake_build(onnx, output, precision, max_batch, opt_batch=1):
        builds.append((onnx, output))
        output.write_bytes(b"engine")
        return output

    monkeypatch.setattr(cli, "_build_engine", fake_build)
    return tmp_path, builds


def onnx_file(d: Path, name: str, body: bytes, sidecar: str | None = None) -> Path:
    p = d / name
    p.write_bytes(body)
    if sidecar is not None:
        p.with_suffix(".json").write_text(sidecar)
    return p


def test_fingerprint_tracks_every_input(env):
    d, _ = env
    a = onnx_file(d, "a.onnx", b"graph-a", '{"num_classes": 80}')
    base = cli._artifact_fingerprint(a, 1, 8)
    assert base == cli._artifact_fingerprint(a, 1, 8)  # deterministic
    onnx_file(d, "a.onnx", b"graph-A", '{"num_classes": 80}')
    assert cli._artifact_fingerprint(a, 1, 8) != base  # bytes changed
    onnx_file(d, "a.onnx", b"graph-a", '{"num_classes": 3}')
    assert cli._artifact_fingerprint(a, 1, 8) != base  # sidecar changed
    onnx_file(d, "a.onnx", b"graph-a", '{"num_classes": 80}')
    assert cli._artifact_fingerprint(a, 8, 8) != base  # profile changed
    assert cli._artifact_fingerprint(a, 1, 8) == base  # and back


def test_explicit_onnx_never_gets_a_foreign_engine(env):
    d, builds = env
    coco = onnx_file(d, "dfine_m_slim.onnx", b"coco-80")
    fresh = onnx_file(d, "custom.onnx", b"custom-3")
    # A cached engine exists — but it was built from the COCO export.
    fp_coco = cli._artifact_fingerprint(coco, 1, 8)
    cli._cache_engine_path("m", "fp16", fp_coco, 1, 8).write_bytes(b"coco engine")

    # Engine-only resolution for the custom ONNX must refuse, not serve COCO.
    with pytest.raises(SystemExit, match="no engine built from custom.onnx"):
        cli._resolve_engine("m", None, "fp16", str(fresh), allow_build=False)
    # With building allowed it builds from the EXPLICIT onnx, into its own slot.
    out = cli._resolve_engine("m", None, "fp16", str(fresh), allow_build=True)
    assert builds == [(fresh, out)]
    assert cli._artifact_fingerprint(fresh, 1, 8) in out.name


def test_rebuilt_export_invalidates_the_old_entry(env):
    d, builds = env
    a = onnx_file(d, "dfine_m_slim.onnx", b"v1")
    first = cli._resolve_engine("m", None, "fp16", None, allow_build=True)
    assert len(builds) == 1
    # Same bytes: the cached engine is reused, no rebuild.
    assert cli._resolve_engine("m", None, "fp16", None, allow_build=True) == first
    assert len(builds) == 1
    # Re-exported (different bytes): a new identity, a fresh build.
    onnx_file(d, "dfine_m_slim.onnx", b"v2")
    second = cli._resolve_engine("m", None, "fp16", None, allow_build=True)
    assert second != first and len(builds) == 2


def test_orphaned_engines_are_never_picked_silently(env, capsys):
    d, _ = env
    # One orphan (its ONNX is gone): usable, with a warning.
    orphan = cli._cache_engine_path("m", "fp16", "aaaaaaaaaaaa", 1, 8)
    orphan.write_bytes(b"e")
    assert cli._resolve_engine("m", None, "fp16", None, allow_build=False) == orphan
    assert "provenance" in capsys.readouterr().err
    # Two orphans: ambiguous — refuse instead of guessing.
    cli._cache_engine_path("m", "fp16", "bbbbbbbbbbbb", 1, 8).write_bytes(b"e")
    with pytest.raises(SystemExit, match="several cached engines"):
        cli._resolve_engine("m", None, "fp16", None, allow_build=False)


def test_legacy_cache_entry_is_a_warned_fallback(env, capsys):
    d, _ = env
    legacy = cli._legacy_cache_engine_path("m", "fp16")
    legacy.write_bytes(b"e")
    assert cli._resolve_engine("m", None, "fp16", None, allow_build=False) == legacy
    assert "pre-v0.3.1" in capsys.readouterr().err


def test_info_and_bench_have_no_dead_onnx_flag():
    parser = cli.build_parser()
    for sub in ("info", "bench"):
        with pytest.raises(SystemExit):
            parser.parse_args([sub, "--model", "m", "--onnx", "x.onnx"])
