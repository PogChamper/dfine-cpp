from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "trt-files/scripts/preset_sweep.py"
SPEC = importlib.util.spec_from_file_location("preset_sweep", SCRIPT)
SWEEP = importlib.util.module_from_spec(SPEC)
sys.modules["preset_sweep"] = SWEEP  # dataclasses resolve annotations via sys.modules
sys.path.insert(0, str(SCRIPT.parent))
try:
    SPEC.loader.exec_module(SWEEP)
finally:
    sys.path.remove(str(SCRIPT.parent))


# ------------------------------- point grammar -----------------------------------


def test_named_presets_prefill_graphs():
    fast = SWEEP.parse_point("fast")
    assert dict(fast.graph) == {"queries": 200, "cascade": "1:100"}
    assert fast.precision == "slim"
    assert fast.profile == (1, 1, 8)
    assert fast.mode == "enqueue"

    maximum = SWEEP.parse_point("max:mode=graph,profile=1/8/8")
    assert dict(maximum.graph) == {"queries": 200, "cascade": "1:100", "eval-idx": 2}
    assert maximum.mode == "graph"
    assert maximum.profile == (1, 8, 8)


def test_custom_points_require_explicit_graph_keys():
    custom = SWEEP.parse_point("q150c75:queries=150,cascade=1:75")
    assert dict(custom.graph) == {"queries": 150, "cascade": "1:75"}

    with pytest.raises(SystemExit, match="unknown preset"):
        SWEEP.parse_point("typo")
    with pytest.raises(SystemExit, match="not overridable"):
        SWEEP.parse_point("fast:queries=100")


def test_graph_key_copies_a_preset_under_a_new_name():
    """Sweeping the same graph twice (e.g. slim vs fp32) needs distinct names."""
    fp32 = SWEEP.parse_point("base-fp32:graph=base,precision=fp32")
    assert dict(fp32.graph) == {}
    assert fp32.precision == "fp32"
    assert fp32.eval_key != SWEEP.parse_point("base").eval_key  # precision differs
    assert fp32.graph_key == SWEEP.parse_point("base").graph_key  # export shared

    with pytest.raises(SystemExit, match="must name a preset"):
        SWEEP.parse_point("x:graph=typo")
    with pytest.raises(SystemExit, match="exclusive"):
        SWEEP.parse_point("x:graph=base,queries=100")
    with pytest.raises(SystemExit, match="not overridable"):
        SWEEP.parse_point("fast:graph=base")


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        ("Fast", "point name"),
        ("base:profile=8/1/8", "1 <= min <= opt <= max"),
        ("base:profile=1/8", "MIN/OPT/MAX"),
        ("base:mode=turbo", "mode must be"),
        ("base:precision=int8", "precision must be"),
        ("base:mode=graph,mode=graph", "duplicate point option"),
        ("base:speed=9", "unknown point options"),
        ("x:queries=0", "positive integer"),
        ("x:cascade=1-100", "K:KEEP"),
    ],
)
def test_point_grammar_rejects_bad_specs(spec, match):
    with pytest.raises(SystemExit, match=match):
        SWEEP.parse_point(spec)


def test_point_keys_factor_accuracy_from_throughput():
    fast = SWEEP.parse_point("fast")
    serving = SWEEP.parse_point("fast-serving:queries=200,cascade=1:100,profile=1/8/8")
    stream = SWEEP.parse_point("fast-stream:queries=200,cascade=1:100,mode=graph")
    c100 = SWEEP.parse_point("c100")

    # Same graph/precision -> shared export, convert, and accuracy evaluation.
    assert fast.eval_key == serving.eval_key == stream.eval_key
    # Different profile or mode -> distinct engine builds and latency runs.
    assert len({fast.build_key, serving.build_key, stream.build_key}) == 3
    # C300->100 and Q200->C100 share the cascade string but not the graph.
    assert c100.eval_key != fast.eval_key


# ------------------------------- argument contract --------------------------------


def _args(**overrides):
    defaults = dict(
        model_name="s",
        num_classes=2,
        checkpoint="model.pt",
        class_names=None,
        images="images",
        ann="test.json",
        yolo_dataset=None,
        split="val",
        point=[],
        baseline="base",
        ap_budget=None,
        no_accuracy=False,
        no_latency=True,
        sample_image=None,
        batches=[1, 8],
        warmup=50,
        iters=500,
        rounds=3,
        keep_going=False,
        rebench=False,
        strict_idle=False,
        ld_library_path="",
        out="sweep",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_validate_args_rejects_incoherent_requests(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"ckpt")
    monkeypatch.setattr(SWEEP, "NATIVE_BENCH", tmp_path / "missing-bench")

    with pytest.raises(SystemExit, match="nothing to do"):
        SWEEP.validate_args(_args(no_accuracy=True, no_latency=True))
    with pytest.raises(SystemExit, match="exactly one of"):
        SWEEP.validate_args(_args(yolo_dataset="dir", ann="test.json"))
    with pytest.raises(SystemExit, match="exactly one of"):
        SWEEP.validate_args(_args(ann=None))
    with pytest.raises(SystemExit, match="requires --images"):
        SWEEP.validate_args(_args(images=None, checkpoint=str(checkpoint)))
    with pytest.raises(SystemExit, match="checkpoint does not exist"):
        SWEEP.validate_args(_args())
    with pytest.raises(SystemExit, match="needs accuracy"):
        SWEEP.validate_args(
            _args(checkpoint=str(checkpoint), ap_budget=0.3, no_accuracy=True, no_latency=False)
        )
    with pytest.raises(SystemExit, match="not built"):
        SWEEP.validate_args(
            _args(checkpoint=str(checkpoint), no_latency=False, ld_library_path="x")
        )

    bench = tmp_path / "dfine_bench"
    bench.write_bytes(b"bench")
    monkeypatch.setattr(SWEEP, "NATIVE_BENCH", bench)
    with pytest.raises(SystemExit, match="libnvinfer"):
        SWEEP.validate_args(
            _args(checkpoint=str(checkpoint), no_latency=False, ld_library_path="")
        )


def test_resolve_points_requires_the_baseline():
    args = _args(point=["q200", "fast"])
    with pytest.raises(SystemExit, match="baseline point"):
        SWEEP.resolve_points(args)

    args = _args(point=["base", "base"])
    with pytest.raises(SystemExit, match="duplicate point names"):
        SWEEP.resolve_points(args)

    points, baseline = SWEEP.resolve_points(_args(point=[]))
    assert [point.name for point in points] == list(SWEEP.DEFAULT_POINTS)
    assert baseline.name == "base"


# ------------------------------- planning + caching -------------------------------


class _Recorder:
    def __init__(self):
        self.steps = []
        self.failures = {}

    def run(self, label, argv, cached):
        self.steps.append((label, [str(item) for item in argv], cached))

    def guarded(self, key, label, argv, cached):
        if key in self.failures:
            return False
        self.run(label, argv, cached)
        return True

    def inherit(self, key, *upstream):
        for source in upstream:
            if source in self.failures:
                self.failures.setdefault(key, self.failures[source])
                return True
        return False


def test_exports_deduplicate_shared_graphs(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"ckpt")
    args = _args(checkpoint=str(checkpoint), out=tmp_path / "sweep")
    points = [
        SWEEP.parse_point("fast"),
        SWEEP.parse_point("fast-serving:queries=200,cascade=1:100,profile=1/8/8"),
        SWEEP.parse_point("base"),
    ]
    recorder = _Recorder()

    exports = SWEEP.run_exports(args, points, recorder)

    assert len(exports) == 2  # fast + fast-serving share one graph
    labels = [label for label, _, _ in recorder.steps]
    assert len(labels) == 2
    fast_argv = next(argv for label, argv, _ in recorder.steps if "q200" in label)
    assert "--num-queries" in fast_argv and "--cascade" in fast_argv
    assert all(cached is False for _, _, cached in recorder.steps)


def test_export_cache_requires_matching_sidecar(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"ckpt")
    args = _args(checkpoint=str(checkpoint), out=tmp_path / "sweep")
    point = SWEEP.parse_point("q200")
    onnx = args.out / "graphs" / point.graph_key / "model_fp32.onnx"
    onnx.parent.mkdir(parents=True)
    onnx.write_bytes(b"onnx")
    sidecar = {
        "checkpoint_sha256": SWEEP.sha256_file(checkpoint),
        "num_queries": 200,
        "num_classes": 2,
        "variant": "s",
        "cascade": None,
        "cascade_initial_queries": None,
    }
    onnx.with_suffix(".json").write_text(json.dumps(sidecar))

    recorder = _Recorder()
    SWEEP.run_exports(args, [point], recorder)
    assert recorder.steps[0][2] is True  # matching sidecar -> reused

    sidecar["checkpoint_sha256"] = "0" * 64
    onnx.with_suffix(".json").write_text(json.dumps(sidecar))
    recorder = _Recorder()
    SWEEP.run_exports(args, [point], recorder)
    assert recorder.steps[0][2] is False  # different checkpoint -> re-export


def test_builds_key_profile_and_mode(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"ckpt")
    args = _args(checkpoint=str(checkpoint), out=tmp_path / "sweep")
    points = [
        SWEEP.parse_point("fast"),
        SWEEP.parse_point("fast-stream:queries=200,cascade=1:100,mode=graph"),
    ]
    artifacts = {point.eval_key: tmp_path / "model_slim.onnx" for point in points}
    artifacts[points[0].eval_key].write_bytes(b"slim")

    recorder = _Recorder()
    engines = SWEEP.run_builds(args, points, artifacts, recorder)

    assert len(engines) == 2
    graph_argv = next(argv for label, argv, _ in recorder.steps if label.endswith("graph"))
    enqueue_argv = next(argv for label, argv, _ in recorder.steps if label.endswith("enqueue"))
    assert "--cuda-graph" in graph_argv
    assert "--cuda-graph" not in enqueue_argv


def test_eval_prefers_the_default_profile_engine(tmp_path):
    fast = SWEEP.parse_point("fast")
    serving = SWEEP.parse_point("fast-serving:queries=200,cascade=1:100,profile=1/8/8")
    engines = {
        fast.build_key: Path("default.engine"),
        serving.build_key: Path("serving.engine"),
    }
    chosen, key = SWEEP.eval_engine_for([serving, fast], serving, engines, {})
    assert chosen == Path("default.engine")
    assert key == fast.build_key

    # A failed default-profile build falls back to a surviving sibling.
    chosen, key = SWEEP.eval_engine_for(
        [serving, fast], serving, engines, {fast.build_key: "010-build"}
    )
    assert chosen == Path("serving.engine")
    assert key == serving.build_key


def test_guarded_aborts_by_default_and_marks_under_keep_going(tmp_path, monkeypatch):
    def broken(argv, stdout, stderr):
        stdout.write("boom: engine refused\n")
        return SimpleNamespace(returncode=3)

    monkeypatch.setattr(SWEEP.subprocess, "run", broken)

    strict = SWEEP.Runner(out=tmp_path, keep_going=False)
    with pytest.raises(SystemExit, match="boom: engine refused"):
        strict.guarded("unit", "build.unit", ["tool"], cached=False)

    tolerant = SWEEP.Runner(out=tmp_path, keep_going=True)
    assert tolerant.guarded("unit", "build.unit", ["tool"], cached=False) is False
    assert tolerant.failures == {"unit": "001-build.unit"}
    # A marked unit never re-runs, and dependents inherit the same failed step.
    assert tolerant.guarded("unit", "build.unit", ["tool"], cached=False) is False
    assert tolerant.inherit("downstream", "unit") is True
    assert tolerant.failures["downstream"] == "001-build.unit"


# ------------------------------- table ------------------------------------------


def _row(name, delta_ap, b8):
    return {
        "point": name,
        "graph": "(base)",
        "precision": "slim",
        "profile": "1/1/8",
        "mode": "enqueue",
        "ap": 55.0,
        "delta": {"AP": delta_ap, "APs": None, "AR100": None, "ARs": None},
        "throughput": {"8": {"img_per_s": b8, "range": [b8 - 1, b8 + 1]}},
    }


def test_markdown_marks_budget_and_recommends_fastest(tmp_path):
    args = _args(ap_budget=0.3, batches=[8], out=tmp_path)
    baseline = SWEEP.parse_point("base")
    rows = [
        {**_row("base", None, 520.0), "delta": {}},
        _row("q200", -0.2, 550.0),
        _row("fast", -0.5, 580.0),
        _row("gain", +0.1, 530.0),
    ]

    table = SWEEP.render_markdown(rows, args, baseline)

    lines = {line.split("|")[1].strip(): line for line in table.splitlines() if line.startswith("| `")}
    assert lines["`q200`"].rstrip("| ").endswith("yes")
    assert lines["`fast`"].rstrip("| ").endswith("no")
    assert lines["`gain`"].rstrip("| ").endswith("yes")  # an AP gain is always in budget
    assert "Fastest point within the 0.3 AP budget" in table
    assert "`q200`" in table.splitlines()[-1]
