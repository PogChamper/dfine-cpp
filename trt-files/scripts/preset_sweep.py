#!/usr/bin/env python3
"""Sweep a checkpoint across operating points and print one decision table.

One command runs the documented conversion/validation workflow for every
requested operating point — export, surgical `slim` conversion, engine build,
accuracy on the user's COCO-format split, baseline deltas, and native
throughput — and assembles the results into `sweep.md` / `sweep.json`.

The sweep contains no model, evaluation, or measurement logic of its own:
every step is a subprocess of the corresponding tool (`export_dfine_onnx`,
`convert_fp16_surgical`, `build_engine`, `coco_eval`, `accuracy_chain`,
`profile`, `yolo_to_coco`), so each artifact and report on disk is exactly
what the manual workflow in docs/VALIDATION.md produces, each carries its own
sidecar/contract, and every argv is logged for by-hand reproduction.

An operating point factors as graph x precision x profile x mode. Accuracy
depends only on graph x precision; throughput on all four axes — the planner
deduplicates shared work accordingly.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from evaluation_report import environment_metadata, sha256_file

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parents[1]
NATIVE_BENCH = REPO / "build" / "dfine_bench"

# Named graph presets, as published in docs/BENCHMARKS.md.
PRESET_GRAPHS = {
    "base": {},
    "q200": {"queries": 200},
    "c150": {"cascade": "1:150"},
    "c100": {"cascade": "1:100"},
    "fast": {"queries": 200, "cascade": "1:100"},
    "max": {"queries": 200, "cascade": "1:100", "eval-idx": 2},
}
DEFAULT_POINTS = ("base", "q200", "c150", "c100", "fast")
DEFAULT_PROFILE = (1, 1, 8)
POINT_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
GRAPH_KEYS = ("queries", "cascade", "eval-idx")
IDLE_UTILIZATION_MAX = 3  # percent; matches the published measurement protocol


def _fail(message: str):
    raise SystemExit(f"[sweep]: {message}")


@dataclass(frozen=True)
class Point:
    name: str
    graph: tuple  # sorted (key, value) pairs from GRAPH_KEYS
    precision: str  # slim | fp32
    profile: tuple  # (min, opt, max)
    mode: str  # enqueue | graph

    @property
    def graph_key(self) -> str:
        graph = dict(self.graph)
        queries = graph.get("queries", 300)
        cascade = str(graph.get("cascade", "none")).replace(":", "-")
        eval_idx = graph.get("eval-idx", "last")
        return f"q{queries}_c{cascade}_e{eval_idx}"

    @property
    def eval_key(self) -> str:
        return f"{self.graph_key}_{self.precision}"

    @property
    def build_key(self) -> str:
        minimum, optimum, maximum = self.profile
        return f"{self.eval_key}_{minimum}-{optimum}-{maximum}_{self.mode}"


def parse_profile(value: str) -> tuple:
    parts = value.split("/")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        _fail(f"profile must be MIN/OPT/MAX integers, got {value!r}")
    minimum, optimum, maximum = (int(part) for part in parts)
    if not 1 <= minimum <= optimum <= maximum:
        _fail(f"profile must satisfy 1 <= min <= opt <= max, got {value!r}")
    return minimum, optimum, maximum


def parse_point(spec: str) -> Point:
    name, _, options_text = spec.partition(":")
    if not POINT_NAME.fullmatch(name):
        _fail(f"point name must match {POINT_NAME.pattern}: {name!r}")
    options = {}
    if options_text:
        for item in options_text.split(","):
            key, separator, value = item.partition("=")
            if not separator or not key or not value:
                _fail(f"point option must be key=value: {item!r} in {spec!r}")
            if key in options:
                _fail(f"duplicate point option {key!r} in {spec!r}")
            options[key] = value

    base_graph = options.pop("graph", None)
    graph_options = {key: options.pop(key) for key in GRAPH_KEYS if key in options}
    if name in PRESET_GRAPHS:
        if graph_options or base_graph:
            _fail(f"{name!r} is a named preset; graph keys are not overridable")
        graph = dict(PRESET_GRAPHS[name])
    elif base_graph is not None:
        # A renamed copy of a preset graph, e.g. base-fp32:graph=base,precision=fp32.
        if base_graph not in PRESET_GRAPHS:
            _fail(f"graph= must name a preset ({', '.join(PRESET_GRAPHS)}), got {base_graph!r}")
        if graph_options:
            _fail(f"graph= and explicit graph keys are exclusive in {spec!r}")
        graph = dict(PRESET_GRAPHS[base_graph])
    else:
        if not graph_options:
            known = ", ".join(PRESET_GRAPHS)
            _fail(f"unknown preset {name!r}; custom points need graph keys ({known} are named)")
        graph = {}
        if "queries" in graph_options:
            if not graph_options["queries"].isdigit() or int(graph_options["queries"]) <= 0:
                _fail(f"queries must be a positive integer in {spec!r}")
            graph["queries"] = int(graph_options["queries"])
        if "cascade" in graph_options:
            layer, separator, keep = graph_options["cascade"].partition(":")
            if not separator or not layer.isdigit() or not keep.isdigit():
                _fail(f"cascade must be K:KEEP in {spec!r}")
            graph["cascade"] = graph_options["cascade"]
        if "eval-idx" in graph_options:
            if not graph_options["eval-idx"].isdigit():
                _fail(f"eval-idx must be a non-negative integer in {spec!r}")
            graph["eval-idx"] = int(graph_options["eval-idx"])

    precision = options.pop("precision", "slim")
    if precision not in ("slim", "fp32"):
        _fail(f"precision must be slim or fp32 in {spec!r}")
    mode = options.pop("mode", "enqueue")
    if mode not in ("enqueue", "graph"):
        _fail(f"mode must be enqueue or graph in {spec!r}")
    profile = parse_profile(options.pop("profile", "1/1/8"))
    if options:
        _fail(f"unknown point options {sorted(options)} in {spec!r}")
    return Point(
        name=name,
        graph=tuple(sorted(graph.items())),
        precision=precision,
        profile=profile,
        mode=mode,
    )


# ------------------------------- step running -----------------------------------


class StepFailure(RuntimeError):
    """One pipeline step failed; the sweep decides whether the run continues."""


@dataclass
class Runner:
    out: Path
    keep_going: bool = False
    log: list = field(default_factory=list)
    failures: dict = field(default_factory=dict)  # unit key -> failed step name
    index: int = 0

    def run(self, label: str, argv: list, cached: bool) -> None:
        self.index += 1
        entry = {
            "step": f"{self.index:03d}-{label}",
            "argv": [str(item) for item in argv],
            "cached": cached,
        }
        self.log.append(entry)
        if cached:
            print(f"[sweep] {entry['step']}: reused existing output")
            return
        log_path = self.out / "logs" / f"{entry['step']}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[sweep] {entry['step']}: {shlex.join(entry['argv'])}")
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(entry["argv"], stdout=stream, stderr=subprocess.STDOUT)
        entry["seconds"] = round(time.monotonic() - started, 1)
        if completed.returncode != 0:
            tail = "".join(log_path.read_text().splitlines(keepends=True)[-15:])
            raise StepFailure(
                f"step {entry['step']} failed (exit {completed.returncode}); "
                f"full log: {log_path}\n{tail}"
            )

    def guarded(self, key: str, label: str, argv: list, cached: bool) -> bool:
        """Run one unit; under --keep-going a failure marks `key` and continues.

        Returns True when the unit's output is (still) trustworthy.
        """
        if key in self.failures:
            return False
        try:
            self.run(label, argv, cached)
        except StepFailure as failure:
            if not self.keep_going:
                _fail(str(failure))
            print(f"[sweep] continuing past {key}: {failure}".splitlines()[0])
            self.failures[key] = f"{self.index:03d}-{label}"
            return False
        return True

    def inherit(self, key: str, *upstream: str) -> bool:
        """Propagate an upstream failure to `key`; True when any upstream failed."""
        for source in upstream:
            if source in self.failures:
                self.failures.setdefault(key, self.failures[source])
                return True
        return False


def _tool(name: str) -> list:
    return [sys.executable, str(SCRIPTS / name)]


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _sidecar_matches(sidecar: Path, expected: dict) -> bool:
    meta = _read_json(sidecar)
    return bool(meta) and all(meta.get(key) == value for key, value in expected.items())


# ------------------------------- planning ---------------------------------------


def graph_export_args(graph: tuple) -> list:
    arguments = []
    options = dict(graph)
    if "queries" in options:
        arguments += ["--num-queries", str(options["queries"])]
    if "cascade" in options:
        arguments += ["--cascade", options["cascade"]]
    if "eval-idx" in options:
        arguments += ["--eval-idx", str(options["eval-idx"])]
    return arguments


def run_exports(args, points: list, runner: Runner) -> dict:
    checkpoint_sha = sha256_file(args.checkpoint)
    exports = {}
    for point in points:
        if point.graph_key in exports:
            continue
        onnx = args.out / "graphs" / point.graph_key / "model_fp32.onnx"
        exports[point.graph_key] = onnx
        graph = dict(point.graph)
        # The sidecar's query contract: with a cascade, num_queries is the KEEP
        # count and cascade_initial_queries the pre-prune count.
        cascade = graph.get("cascade")
        output_queries = int(cascade.split(":")[1]) if cascade else graph.get("queries", 300)
        expected = {
            "checkpoint_sha256": checkpoint_sha,
            "num_queries": output_queries,
            "num_classes": args.num_classes,
            "variant": args.model_name,
            "cascade": cascade,
            "cascade_initial_queries": graph.get("queries", 300) if cascade else None,
        }
        if "eval-idx" in graph:
            expected["eval_idx"] = graph["eval-idx"]
        cached = onnx.is_file() and _sidecar_matches(onnx.with_suffix(".json"), expected)
        argv = (
            _tool("export_dfine_onnx.py")
            + ["--model-name", args.model_name, "--checkpoint", args.checkpoint]
            + ["--opset", "19", "--num-classes", str(args.num_classes)]
            + (["--class-names", args.class_names] if args.class_names else [])
            + graph_export_args(point.graph)
            + ["--output", str(onnx)]
        )
        runner.guarded(point.graph_key, f"export.{point.graph_key}", argv, cached)
    return exports


def run_converts(args, points: list, exports: dict, runner: Runner) -> dict:
    artifacts = {}
    for point in points:
        if point.eval_key in artifacts:
            continue
        fp32 = exports[point.graph_key]
        if runner.inherit(point.eval_key, point.graph_key):
            artifacts[point.eval_key] = fp32
            continue
        if point.precision == "fp32":
            artifacts[point.eval_key] = fp32
            continue
        slim = fp32.parent / "model_slim.onnx"
        artifacts[point.eval_key] = slim
        cached = slim.is_file() and _sidecar_matches(
            slim.with_suffix(".json"), {"source_onnx_sha256": sha256_file(fp32)}
        )
        argv = _tool("convert_fp16_surgical.py") + [
            "--onnx",
            str(fp32),
            "--output",
            str(slim),
            "--slim",
        ]
        runner.guarded(point.eval_key, f"convert.{point.eval_key}", argv, cached)
    return artifacts


def run_builds(args, points: list, artifacts: dict, runner: Runner) -> dict:
    engines = {}
    for point in points:
        if point.build_key in engines:
            continue
        onnx = artifacts[point.eval_key]
        engine = args.out / "engines" / point.build_key / "model.engine"
        engines[point.build_key] = engine
        minimum, optimum, maximum = point.profile
        expected = {
            "onnx_sha256": sha256_file(onnx) if onnx.is_file() else "missing",
            "min_batch": minimum,
            "opt_batch": optimum,
            "max_batch": maximum,
            "cuda_graph_compat": point.mode == "graph",
        }
        if runner.inherit(point.build_key, point.eval_key):
            continue
        cached = engine.is_file() and _sidecar_matches(engine.with_suffix(".json"), expected)
        argv = (
            _tool("build_engine.py")
            + ["--onnx", str(onnx), "--output", str(engine)]
            + ["--strongly-typed", "--no-tf32"]
            + ["--min-batch", str(minimum), "--opt-batch", str(optimum)]
            + ["--max-batch", str(maximum)]
            + (["--cuda-graph"] if point.mode == "graph" else [])
        )
        runner.guarded(point.build_key, f"build.{point.build_key}", argv, cached)
    return engines


def eval_engine_for(points: list, target: Point, engines: dict, failures: dict) -> tuple:
    """Accuracy is profile/mode-independent; prefer the default-profile enqueue build."""
    candidates = [point for point in points if point.eval_key == target.eval_key]
    candidates.sort(
        key=lambda point: (
            point.build_key in failures,
            not (point.profile == DEFAULT_PROFILE and point.mode == "enqueue"),
        )
    )
    chosen = candidates[0]
    return engines[chosen.build_key], chosen.build_key


def run_evals(args, points: list, artifacts: dict, engines: dict, runner: Runner) -> dict:
    annotations_sha = sha256_file(args.ann)
    reports = {}
    for point in points:
        if point.eval_key in reports:
            continue
        engine, engine_key = eval_engine_for(points, point, engines, runner.failures)
        report = args.out / "reports" / point.eval_key / "accuracy.json"
        reports[point.eval_key] = report
        if runner.inherit(point.eval_key, engine_key):
            continue
        payload = _read_json(report)
        engine_entry = payload.get("backends", {}).get("engine", {})
        cached = (
            engine_entry.get("artifact", {}).get("sha256") == sha256_file(engine)
            if engine.is_file()
            else False
        ) and payload.get("evaluation_contract", {}).get("annotations_sha256") == annotations_sha
        argv = (
            _tool("coco_eval.py")
            + ["--backends", "onnx", "engine"]
            + ["--model-name", args.model_name, "--num-classes", str(args.num_classes)]
            + ["--onnx", str(artifacts[point.eval_key]), "--engine", str(engine)]
            + ["--images", args.images, "--ann", args.ann]
            + ["--report", str(report), "--overwrite"]
        )
        runner.guarded(point.eval_key, f"eval.{point.eval_key}", argv, cached)
    return reports


def run_chains(args, points: list, baseline: Point, reports: dict, runner: Runner) -> dict:
    chains = {}
    for point in points:
        if point.eval_key == baseline.eval_key or point.eval_key in chains:
            continue
        if baseline.eval_key in runner.failures or point.eval_key in runner.failures:
            continue
        chain = args.out / "reports" / point.eval_key / "vs-baseline.json"
        chains[point.eval_key] = chain
        stages = [
            f"{baseline.name}={reports[baseline.eval_key]}::engine",
            f"{point.name}={reports[point.eval_key]}::engine",
        ]
        argv = (
            _tool("accuracy_chain.py")
            + ["--stage", stages[0], "--stage", stages[1]]
            + ["--transition-kind", "preset"]
            + ["--transition-label", f"{point.name} vs {baseline.name}"]
            + ["--output", str(chain), "--overwrite"]
        )
        # Chains re-read both reports, so a fresh eval invalidates them cheaply.
        runner.guarded(f"{point.eval_key}:chain", f"chain.{point.eval_key}", argv, cached=False)
    return chains


def require_idle_gpu(strict: bool) -> None:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    if query.returncode != 0:
        _fail("nvidia-smi is unavailable; cannot verify the GPU before throughput steps")
    utilization = int(query.stdout.split()[0])
    if utilization > IDLE_UTILIZATION_MAX:
        message = (
            f"GPU utilization is {utilization}% before a throughput step; "
            "concurrent load skews latency (see docs/BENCHMARKS.md protocol)"
        )
        if strict:
            _fail(message)
        print(f"[sweep] warning: {message}")


def run_benches(args, points: list, engines: dict, runner: Runner) -> dict:
    latencies = {}
    for point in points:
        if point.build_key in latencies:
            continue
        engine = engines[point.build_key]
        report = args.out / "reports" / point.build_key / "latency.json"
        latencies[point.build_key] = report
        if point.build_key in runner.failures:
            continue
        payload = _read_json(report)
        backend = "cpp-graph" if point.mode == "graph" else "cpp"
        entry = payload.get("backends", {}).get(backend, {})
        cached = (
            not args.rebench
            and engine.is_file()
            and entry.get("artifact", {}).get("sha256") == sha256_file(engine)
            and all(str(batch) in entry.get("latency", {}) for batch in args.batches)
        )
        if not cached:
            require_idle_gpu(args.strict_idle)
        argv = (
            _tool("profile.py")
            + ["--backends", backend, "--engine", str(engine)]
            + ["--model-name", args.model_name, "--num-classes", str(args.num_classes)]
            + ["--sample-image", args.sample_image]
            + ["--batches"]
            + [str(batch) for batch in args.batches]
            + ["--warmup", str(args.warmup), "--iters", str(args.iters)]
            + ["--rounds", str(args.rounds), "--no-accuracy"]
            + ["--ld-library-path", args.ld_library_path]
            + ["--out", str(report), "--overwrite"]
        )
        runner.guarded(f"{point.build_key}:bench", f"bench.{point.build_key}", argv, cached)
    return latencies


# ------------------------------- table ------------------------------------------


def _round_points(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(value * 100, digits)


def point_row(
    point: Point, reports: dict, chains: dict, latencies: dict, args, failures: dict
) -> dict:
    row = {
        "point": point.name,
        "graph": " ".join(graph_export_args(point.graph)) or "(base)",
        "precision": point.precision,
        "profile": "/".join(str(part) for part in point.profile),
        "mode": point.mode,
    }
    failed = next(
        (
            failures[key]
            for key in (
                point.graph_key,
                point.eval_key,
                point.build_key,
                f"{point.eval_key}:chain",
                f"{point.build_key}:bench",
            )
            if key in failures
        ),
        None,
    )
    if failed:
        row["failed"] = failed
    if point.eval_key in reports:
        accuracy = _read_json(reports[point.eval_key])
        metrics = accuracy.get("backends", {}).get("engine", {}).get("map", {})
        row["ap"] = _round_points(metrics.get("AP"))
    else:
        row["ap"] = None
    chain = _read_json(chains[point.eval_key]) if point.eval_key in chains else {}
    if chain.get("transitions"):
        delta = chain["transitions"][0]["delta"]["bbox"]
        row["delta"] = {
            name: _round_points(delta.get(name)) for name in ("AP", "APs", "AR100", "ARs")
        }
    latency_report = (
        _read_json(latencies[point.build_key]) if point.build_key in latencies else {}
    )
    backend = "cpp-graph" if point.mode == "graph" else "cpp"
    latency = latency_report.get("backends", {}).get(backend, {}).get("latency")
    if latency:
        row["throughput"] = {}
        for batch in args.batches:
            scope = latency[str(batch)]["scopes"]["end_to_end"]
            rounds = [round_result["img_per_s"] for round_result in scope["rounds"]]
            row["throughput"][str(batch)] = {
                "img_per_s": round(scope["img_per_s"], 1),
                "range": [round(min(rounds), 1), round(max(rounds), 1)],
            }
    return row


def render_markdown(rows: list, args, baseline: Point) -> str:
    batches = [str(batch) for batch in args.batches]
    header = (
        ["Point", "Graph", "Profile", "Mode", "AP", "ΔAP", "ΔAPs", "ΔAR100"]
        + [f"b{batch} img/s" for batch in batches]
        + (["In budget"] if args.ap_budget is not None else [])
    )
    lines = [
        "# Sweep results",
        "",
        f"Checkpoint: `{args.checkpoint}`; dataset: `{args.ann}`; baseline: `{baseline.name}`.",
        "Throughput is native C++ image-to-detections (median of "
        f"{args.rounds}×{args.iters} iterations; ranges in sweep.json).",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    budget_rows = []
    failed_rows = [row for row in rows if row.get("failed")]
    for row in rows:
        delta = row.get("delta", {})
        in_budget = None
        if args.ap_budget is not None:
            if row.get("failed"):
                in_budget = False
            else:
                in_budget = (
                    delta.get("AP") is None
                    or abs(min(delta.get("AP", 0.0), 0.0)) <= args.ap_budget
                )
            if in_budget and row.get("throughput"):
                budget_rows.append(row)
        cells = [
            f"`{row['point']}`",
            f"`{row['graph']}`",
            row["profile"],
            row["mode"],
            "—" if row["ap"] is None else f"{row['ap']:.3f}",
            *(
                "—" if delta.get(name) is None else f"{delta[name]:+.3f}"
                for name in ("AP", "APs", "AR100")
            ),
            *(
                (
                    f"{row['throughput'][batch]['img_per_s']:.0f}"
                    if row.get("throughput", {}).get(batch)
                    else "—"
                )
                for batch in batches
            ),
        ]
        if args.ap_budget is not None:
            cells.append("yes" if in_budget else "no")
        lines.append("| " + " | ".join(cells) + " |")
    if args.ap_budget is not None and budget_rows:
        best_batch = batches[-1]
        best = max(budget_rows, key=lambda row: row["throughput"][best_batch]["img_per_s"])
        lines += [
            "",
            f"Fastest point within the {args.ap_budget} AP budget at batch {best_batch}: "
            f"`{best['point']}` "
            f"({best['throughput'][best_batch]['img_per_s']:.0f} img/s, "
            f"ΔAP {best.get('delta', {}).get('AP', 0.0):+.3f}).",
        ]
    if failed_rows:
        lines += [""] + [
            f"`{row['point']}` failed at step {row['failed']} (see logs/)."
            for row in failed_rows
        ]
    return "\n".join(lines) + "\n"


# ------------------------------- CLI --------------------------------------------


def default_ld_library_path() -> str:
    import importlib.util

    spec = importlib.util.find_spec("tensorrt_libs")
    if spec and spec.submodule_search_locations:
        return str(Path(list(spec.submodule_search_locations)[0]))
    return ""


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep a checkpoint across conversion/optimization operating points"
    )
    parser.add_argument("--model-name", required=True, choices=("n", "s", "m", "l", "x"))
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--class-names", help="file (one per line) or comma list")
    dataset = parser.add_argument_group("dataset (COCO json, or a YOLO dir to convert)")
    dataset.add_argument("--images", help="image directory for --ann")
    dataset.add_argument("--ann", help="COCO-format annotations json")
    dataset.add_argument("--yolo-dataset", help="D-FINE-seg layout: images/, labels/, <split>.csv")
    dataset.add_argument("--split", default="val", help="YOLO split to convert (default: val)")
    parser.add_argument(
        "--point",
        action="append",
        default=[],
        metavar="NAME[:k=v,...]",
        help="operating point; keys: queries, cascade, eval-idx, graph=<preset>, "
        "profile=MIN/OPT/MAX, mode=enqueue|graph, precision=slim|fp32 "
        f"(default points: {','.join(DEFAULT_POINTS)})",
    )
    parser.add_argument("--baseline", default="base", help="delta anchor point name")
    parser.add_argument("--ap-budget", type=float, help="max acceptable AP drop vs baseline, in points")
    parser.add_argument("--no-accuracy", action="store_true")
    parser.add_argument("--no-latency", action="store_true")
    parser.add_argument("--sample-image", help="latency input (default: first image of the split)")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="continue past a failed point (its row is marked); default aborts on first failure",
    )
    parser.add_argument("--rebench", action="store_true", help="ignore cached latency reports")
    parser.add_argument("--strict-idle", action="store_true", help="refuse a busy GPU instead of warning")
    parser.add_argument("--ld-library-path", default=default_ld_library_path())
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def validate_args(args) -> None:
    if args.no_accuracy and args.no_latency:
        _fail("nothing to do: both accuracy and latency are disabled")
    if bool(args.yolo_dataset) == bool(args.ann):
        _fail("pass exactly one of --ann (with --images) or --yolo-dataset")
    if args.ann and not args.images:
        _fail("--ann requires --images")
    if not Path(args.checkpoint).is_file():
        _fail(f"checkpoint does not exist: {args.checkpoint}")
    if args.ap_budget is not None and args.ap_budget < 0:
        _fail("--ap-budget must be non-negative")
    if args.ap_budget is not None and args.no_accuracy:
        _fail("--ap-budget needs accuracy; drop --no-accuracy")
    if not args.no_latency:
        if not NATIVE_BENCH.is_file():
            _fail(f"{NATIVE_BENCH} is not built; run ./build.sh (or pass --no-latency)")
        if not args.ld_library_path:
            _fail("cannot locate libnvinfer; pass --ld-library-path")


def resolve_points(args) -> tuple:
    specs = args.point or list(DEFAULT_POINTS)
    points = [parse_point(spec) for spec in specs]
    names = [point.name for point in points]
    if len(names) != len(set(names)):
        _fail(f"duplicate point names: {sorted(names)}")
    baseline = next((point for point in points if point.name == args.baseline), None)
    if baseline is None and not args.no_accuracy:
        _fail(
            f"baseline point {args.baseline!r} is not in the sweep; deltas and budgets "
            "need it (add it, or pick another --baseline)"
        )
    return points, baseline


def resolve_dataset(args, runner: Runner) -> None:
    if args.yolo_dataset:
        if not args.class_names:
            _fail("--yolo-dataset requires --class-names (the training label order)")
        converted = args.out / "dataset" / f"instances_{args.split}.json"
        argv = _tool("yolo_to_coco.py") + [
            "--dataset",
            args.yolo_dataset,
            "--split",
            args.split,
            "--class-names",
            args.class_names,
            "--output",
            str(converted),
            "--overwrite",
        ]
        runner.run(f"dataset.{args.split}", argv, cached=False)
        args.ann = str(converted)
        args.images = str(Path(args.yolo_dataset) / "images")
    if not Path(args.ann).is_file():
        _fail(f"annotations do not exist: {args.ann}")
    if not Path(args.images).is_dir():
        _fail(f"image directory does not exist: {args.images}")
    if args.sample_image is None:
        annotations = _read_json(Path(args.ann))
        names = sorted(image["file_name"] for image in annotations.get("images", []))
        if not names:
            _fail(f"annotations list no images: {args.ann}")
        args.sample_image = str(Path(args.images) / names[0])
    if not args.no_latency and not Path(args.sample_image).is_file():
        _fail(f"sample image does not exist: {args.sample_image}")


def main(args) -> int:
    validate_args(args)
    args.out = Path(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    runner = Runner(out=args.out, keep_going=args.keep_going)
    points, baseline = resolve_points(args)
    resolve_dataset(args, runner)

    exports = run_exports(args, points, runner)
    artifacts = run_converts(args, points, exports, runner)
    engines = run_builds(args, points, artifacts, runner)
    reports = run_evals(args, points, artifacts, engines, runner) if not args.no_accuracy else {}
    chains = (
        run_chains(args, points, baseline, reports, runner) if not args.no_accuracy else {}
    )
    latencies = run_benches(args, points, engines, runner) if not args.no_latency else {}

    rows = [
        point_row(point, reports, chains, latencies, args, runner.failures) for point in points
    ]
    sweep = {
        "schema_version": 1,
        "checkpoint": {"path": str(args.checkpoint), "sha256": sha256_file(args.checkpoint)},
        "dataset": {
            "images": str(args.images),
            "annotations": str(args.ann),
            "annotations_sha256": sha256_file(args.ann),
        },
        "points": rows,
        "steps": runner.log,
        "failures": runner.failures,
        "environment": environment_metadata(),
    }
    (args.out / "sweep.json").write_text(json.dumps(sweep, indent=2, sort_keys=True) + "\n")
    table = render_markdown(rows, args, baseline or points[0])
    (args.out / "sweep.md").write_text(table)
    print(f"[sweep] wrote {args.out / 'sweep.md'} and sweep.json")
    print()
    print(table)
    return 1 if runner.failures else 0


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
