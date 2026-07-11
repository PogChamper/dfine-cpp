"""Engine-cache identity: an engine is bound to the exact ONNX + sidecar bytes
it was built from (the batch profile is recorded in the engine sidecar and is
NOT identity), an explicit --onnx can never lose to a cache entry built from
something else, and provenance-less fallbacks are never picked silently among
several candidates. Regression for the v0.3.0 shadowing bug (a stale COCO
engine silently served a fresh custom export)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dfine import cli


@pytest.fixture()
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(cli, "_repo_root", lambda: None)  # no dev-tree fallback
    monkeypatch.setattr(cli, "_gpu_arch", lambda: "89")
    monkeypatch.setattr(cli, "_trt_version", lambda: "10.13")
    builds: list[tuple[Path, Path]] = []

    def fake_build(onnx, output, precision, max_batch, opt_batch=1):
        builds.append((onnx, output))
        output.write_bytes(b"engine")
        # Mimic build_engine.py's publish contract: the sidecar lands at the
        # stem name and a stale appended-name twin is removed (readers probe
        # the appended name first, so a leftover would shadow the fresh one).
        twin = Path(str(output) + ".json")
        if twin.exists():
            twin.unlink()
        output.with_suffix(".json").write_text(
            json.dumps(
                {
                    "onnx_sha256": hashlib.sha256(Path(onnx).read_bytes()).hexdigest(),
                    "max_batch": max_batch,
                }
            )
        )
        return output

    monkeypatch.setattr(cli, "_build_engine", fake_build)
    return tmp_path, builds


def onnx_file(d: Path, name: str, body: bytes, sidecar: str | None = None) -> Path:
    p = d / name
    p.write_bytes(body)
    if sidecar is not None:
        p.with_suffix(".json").write_text(sidecar)
    return p


def test_fingerprint_tracks_artifact_content_only(env):
    d, _ = env
    a = onnx_file(d, "a.onnx", b"graph-a", '{"num_classes": 80}')
    base = cli._artifact_fingerprint(a)
    assert base == cli._artifact_fingerprint(a)  # deterministic
    onnx_file(d, "a.onnx", b"graph-A", '{"num_classes": 80}')
    assert cli._artifact_fingerprint(a) != base  # bytes changed
    onnx_file(d, "a.onnx", b"graph-a", '{"num_classes": 3}')
    assert cli._artifact_fingerprint(a) != base  # sidecar changed
    onnx_file(d, "a.onnx", b"graph-a", '{"num_classes": 80}')
    assert cli._artifact_fingerprint(a) == base  # and back


def test_explicit_onnx_never_gets_a_foreign_engine(env):
    d, builds = env
    coco = onnx_file(d, "dfine_m_slim.onnx", b"coco-80")
    fresh = onnx_file(d, "custom.onnx", b"custom-3")
    # A cached engine exists — but it was built from the COCO export.
    fp_coco = cli._artifact_fingerprint(coco)
    cli._cache_engine_path("m", "fp16", fp_coco, 1, 8).write_bytes(b"coco engine")

    # Engine-only resolution for the custom ONNX must refuse, not serve COCO.
    with pytest.raises(SystemExit, match="no engine built from custom.onnx"):
        cli._resolve_engine("m", None, "fp16", str(fresh), allow_build=False)
    # With building allowed it builds from the EXPLICIT onnx, into its own slot.
    out = cli._resolve_engine("m", None, "fp16", str(fresh), allow_build=True)
    assert builds == [(fresh, out)]
    assert cli._artifact_fingerprint(fresh) in out.name


def test_rebuilt_export_invalidates_the_old_entry(env):
    d, builds = env
    onnx_file(d, "dfine_m_slim.onnx", b"v1")
    first = cli._resolve_engine("m", None, "fp16", None, allow_build=True)
    assert len(builds) == 1
    # Same bytes: the cached engine is reused, no rebuild.
    assert cli._resolve_engine("m", None, "fp16", None, allow_build=True) == first
    assert len(builds) == 1
    # Re-exported (different bytes): a new identity, a fresh build.
    onnx_file(d, "dfine_m_slim.onnx", b"v2")
    second = cli._resolve_engine("m", None, "fp16", None, allow_build=True)
    assert second != first and len(builds) == 2


def test_same_artifact_other_profile_is_still_served(env, capsys):
    d, _ = env
    a = onnx_file(d, "dfine_m_slim.onnx", b"graph")
    fp = cli._artifact_fingerprint(a)
    # The user built with a serving profile; predict's defaults (1/8) must still
    # find it — the profile shapes performance, not identity.
    built = cli._cache_engine_path("m", "fp16", fp, 8, 16)
    built.write_bytes(b"engine")
    assert cli._resolve_engine("m", None, "fp16", None, allow_build=False) == built
    assert "alternate build configuration" in capsys.readouterr().err
    # An exact-profile entry (predict's default 1/8) wins over any other...
    exact = cli._cache_engine_path("m", "fp16", fp, 1, 8)
    exact.write_bytes(b"engine")
    assert cli._resolve_engine("m", None, "fp16", None, allow_build=False) == exact
    # ...and among non-exact candidates the pick is by NUMERIC max_batch
    # (16 must beat 8 although "16" < "8" sorts first lexicographically).
    exact.unlink()
    cli._cache_engine_path("m", "fp16", fp, 2, 8).write_bytes(b"engine")
    assert cli._resolve_engine("m", None, "fp16", None, allow_build=False) == built


def test_cuda_graph_policy_has_a_distinct_cache_entry(env):
    d, _ = env
    onnx = onnx_file(d, "dfine_m_slim.onnx", b"graph")
    fp = cli._artifact_fingerprint(onnx)
    regular = cli._cache_engine_path("m", "fp16", fp, 1, 8)
    graph = cli._cache_engine_path("m", "fp16", fp, 1, 8, cuda_graph=True)

    assert regular != graph
    assert "-b1-1-8-g0-sm89-" in graph.name

    graph.write_bytes(b"engine")
    assert cli._resolve_engine("m", None, "fp16", None, allow_build=False) == graph


def test_build_command_propagates_cuda_graph_policy(env, monkeypatch):
    d, _ = env
    onnx = onnx_file(d, "dfine_m_slim.onnx", b"graph")
    calls = []

    def fake_build(source, output, precision, max_batch, opt_batch=1, cuda_graph=False):
        calls.append((source, output, precision, max_batch, opt_batch, cuda_graph))
        return output

    monkeypatch.setattr(cli, "_build_engine", fake_build)

    assert cli.main(["build", "--model", "m", "--cuda-graph"]) == 0
    assert calls == [
        (
            onnx,
            cli._cache_engine_path(
                "m",
                "fp16",
                cli._artifact_fingerprint(onnx),
                1,
                8,
                cuda_graph=True,
            ),
            "fp16",
            8,
            1,
            True,
        )
    ]


@pytest.mark.parametrize("cuda_graph", [False, True])
def test_engine_builder_forwards_cuda_graph_alias(monkeypatch, tmp_path, cuda_graph):
    onnx = onnx_file(tmp_path, "model.onnx", b"graph")
    engine = tmp_path / "model.engine"
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        engine.write_bytes(b"engine")

    monkeypatch.setattr(cli, "_have_tensorrt", lambda: True)
    monkeypatch.setattr(cli, "_build_engine_script", lambda: tmp_path / "build_engine.py")
    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    assert cli._build_engine(onnx, engine, "fp16", 8, cuda_graph=cuda_graph) == engine
    assert ("--cuda-graph" in calls[0]) is cuda_graph


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
    assert "unfingerprinted" in capsys.readouterr().err


def engine_with_sidecar(path: Path, meta: dict) -> Path:
    path.write_bytes(b"engine")
    Path(str(path) + ".json").write_text(json.dumps(meta))
    return path


def test_profile_pick_trusts_the_sidecar_over_the_name(env):
    d, _ = env
    a = onnx_file(d, "dfine_m_slim.onnx", b"graph")
    fp = cli._artifact_fingerprint(a)
    # Names and sidecars disagree on max_batch; the sidecar wins — the filename
    # is a label, consulted only for engines that predate the sidecar.
    engine_with_sidecar(cli._cache_engine_path("m", "fp16", fp, 1, 16), {"max_batch": 4})
    big = engine_with_sidecar(cli._cache_engine_path("m", "fp16", fp, 2, 4), {"max_batch": 32})
    assert cli._resolve_engine("m", None, "fp16", None, allow_build=False) == big


def test_engine_recorded_from_another_source_is_refused(env, capsys):
    d, builds = env
    a = onnx_file(d, "dfine_m_slim.onnx", b"graph")
    fp = cli._artifact_fingerprint(a)
    fake = engine_with_sidecar(
        cli._cache_engine_path("m", "fp16", fp, 1, 8), {"onnx_sha256": "0" * 64, "max_batch": 8}
    )
    # The filename matches the artifact, the recorded source does not: never served...
    with pytest.raises(SystemExit, match="no engine built from"):
        cli._resolve_engine("m", None, "fp16", None, allow_build=False)
    assert "different source ONNX" in capsys.readouterr().err
    # ...and with building allowed it is rebuilt in place, not trusted.
    out = cli._resolve_engine("m", None, "fp16", None, allow_build=True)
    assert builds[-1] == (a, fake) and out == fake
    # The rebuild replaced the poisoned sidecar, so resolution CONVERGES:
    # the next resolve serves the cache without another build.
    assert cli._resolve_engine("m", None, "fp16", None, allow_build=False) == fake
    assert len(builds) == 1


def test_engine_meta_probes_the_appended_name_first(env):
    d, _ = env
    e = d / "x.engine"
    e.write_bytes(b"engine")
    e.with_suffix(".json").write_text('{"max_batch": 4}')
    assert cli._engine_meta(e) == {"max_batch": 4}
    # The appended name wins, matching the C++ runtime's probe order.
    Path(str(e) + ".json").write_text('{"max_batch": 16}')
    assert cli._engine_meta(e) == {"max_batch": 16}


def test_info_and_bench_have_no_dead_onnx_flag():
    parser = cli.build_parser()
    for sub in ("info", "bench"):
        with pytest.raises(SystemExit):
            parser.parse_args([sub, "--model", "m", "--onnx", "x.onnx"])


@pytest.mark.parametrize("threshold", ["-0.1", "1.1", "nan", "inf"])
def test_predict_parser_rejects_invalid_threshold(threshold):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            [
                "predict",
                "--engine",
                "model.engine",
                "--image",
                "frame.jpg",
                "--threshold",
                threshold,
            ]
        )


@pytest.mark.parametrize(("opt_batch", "max_batch"), [(0, 8), (1, 0), (9, 8)])
def test_engine_build_rejects_invalid_profile_before_tensorrt(
    monkeypatch,
    tmp_path,
    opt_batch,
    max_batch,
):
    def unexpected_probe():
        raise AssertionError("TensorRT must not be probed for an invalid profile")

    monkeypatch.setattr(cli, "_have_tensorrt", unexpected_probe)
    with pytest.raises(SystemExit, match="1 <= opt <= max"):
        cli._build_engine(
            tmp_path / "model.onnx",
            tmp_path / "model.engine",
            "fp16",
            max_batch=max_batch,
            opt_batch=opt_batch,
        )


def test_gpu_arch_uses_cuda_runtime_device(monkeypatch):
    class CudaRuntime:
        @staticmethod
        def cudaGetDevice(device):
            device._obj.value = 0
            return 0

        @staticmethod
        def cudaDeviceGetAttribute(value, attribute, device):
            assert device == 0
            value._obj.value = {75: 8, 76: 6}[attribute]
            return 0

    monkeypatch.setattr(cli.ctypes, "CDLL", lambda _: CudaRuntime())
    assert cli._gpu_arch() == "86"
