"""Strict checkpoint loading in export_dfine_onnx.py: a checkpoint that does not
exactly fill the model's tensors must stop the export (with actionable hints),
not silently export randomly-initialized weights. CPU-only; skips without torch."""

from __future__ import annotations

import argparse
import builtins
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def exporter():
    spec = importlib.util.spec_from_file_location(
        "export_dfine_onnx", REPO / "trt-files/scripts/export_dfine_onnx.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TinyHead(nn.Module):
    """Key layout mimics the D-FINE naming the hint logic keys off
    (dec_score_head.<i>.weight)."""

    def __init__(self, num_classes: int = 80):
        super().__init__()
        self.backbone = nn.Linear(8, 16)
        self.dec_score_head = nn.ModuleList([nn.Linear(16, num_classes)])


def save_ckpt(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "ckpt.pt"
    torch.save(payload, p)
    return p


def test_select_state_prefers_ema(exporter):
    ema = {"w": torch.zeros(1)}
    raw = {"ema": {"module": ema}, "model": {"w": torch.ones(1)}}
    state, name = exporter._select_state(raw)
    assert state is ema and name == "ema.module"
    state, name = exporter._select_state({"model": ema})
    assert state is ema and name == "model"
    state, name = exporter._select_state(ema)
    assert state is ema and name == "checkpoint root"


def test_strict_exact_load(exporter, tmp_path, monkeypatch):
    model = TinyHead()
    ckpt = save_ckpt(tmp_path, {"model": TinyHead().state_dict()})
    reads = 0
    read_bytes = Path.read_bytes

    def counted_read(path):
        nonlocal reads
        reads += 1
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read)
    report = exporter.load_checkpoint_state(model, ckpt, allow_partial=False)
    assert report["mode"] == "strict"
    assert report["loaded"] == len(model.state_dict())
    assert report["missing"] == 0 and report["shape_mismatch"] == 0
    assert len(report["sha256"]) == 64
    assert reads == 1


def test_zero_matching_keys_aborts(exporter, tmp_path):
    ckpt = save_ckpt(tmp_path, {"model": {"unrelated.weight": torch.zeros(3)}})
    with pytest.raises(SystemExit, match="missing"):
        exporter.load_checkpoint_state(TinyHead(), ckpt, allow_partial=False)


def test_class_count_mismatch_hints_num_classes(exporter, tmp_path):
    ckpt = save_ckpt(tmp_path, {"model": TinyHead(num_classes=3).state_dict()})
    with pytest.raises(SystemExit, match=r"--num-classes 3"):
        exporter.load_checkpoint_state(TinyHead(num_classes=80), ckpt, allow_partial=False)


def test_variant_mismatch_hints_model_name(exporter, tmp_path):
    # Nothing fits: every model tensor is missing -> the variant hint fires.
    ckpt = save_ckpt(tmp_path, {"model": {"other.w": torch.zeros(2)}})
    with pytest.raises(SystemExit, match="model-name"):
        exporter.load_checkpoint_state(TinyHead(), ckpt, allow_partial=False)


def test_partial_mode_loads_and_reports(exporter, tmp_path, capsys):
    model = TinyHead(num_classes=80)
    donor = TinyHead(num_classes=3)
    ckpt = save_ckpt(tmp_path, {"model": donor.state_dict()})
    report = exporter.load_checkpoint_state(model, ckpt, allow_partial=True)
    assert report["mode"] == "partial"
    assert report["shape_mismatch"] == 2  # score head weight + bias
    # The backbone (matching shapes) really was loaded.
    assert torch.equal(model.backbone.weight, donor.backbone.weight)
    assert "PARTIAL" in capsys.readouterr().out


def test_extra_checkpoint_tensors_tolerated(exporter, tmp_path, capsys):
    sd = TinyHead().state_dict()
    sd["seg_head.weight"] = torch.zeros(4, 4)  # e.g. a mask head on a detect export
    ckpt = save_ckpt(tmp_path, {"model": sd})
    report = exporter.load_checkpoint_state(TinyHead(), ckpt, allow_partial=False)
    assert report["mode"] == "strict"
    assert "unused" in capsys.readouterr().out


def test_bare_state_dict_checkpoint(exporter, tmp_path):
    ckpt = save_ckpt(tmp_path, TinyHead().state_dict())
    report = exporter.load_checkpoint_state(TinyHead(), ckpt, allow_partial=False)
    assert report["selected_state"] == "checkpoint root"


def test_regenerated_geometry_buffers_retarget_vs_same_size(exporter, tmp_path):
    # decoder.anchors / decoder.valid_mask are size-derived geometry. On an
    # --img-size retarget (shape mismatch) they are ignored — strict-clean.
    model_sd = {"backbone.w": torch.zeros(4), "decoder.anchors": torch.zeros(1, 100, 4)}
    state = {"backbone.w": torch.zeros(4), "decoder.anchors": torch.zeros(1, 400, 4)}
    diff = exporter._diff_state(model_sd, state)
    assert diff["missing"] == [] and diff["shape_mismatch"] == [] and diff["extra"] == []

    # At the training size (shapes match) the CHECKPOINT copies must load —
    # the gated release assets embed exactly those values, so exports stay
    # byte-reproducible.
    class Geo(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("anchors", torch.zeros(1, 10, 4))

    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = Geo()

    model = Wrap()
    donor = Wrap()
    donor.decoder.anchors += 0.5
    ckpt = save_ckpt(tmp_path, {"model": donor.state_dict()})
    report = exporter.load_checkpoint_state(model, ckpt, allow_partial=False)
    assert report["mode"] == "strict"
    assert torch.equal(model.decoder.anchors, donor.decoder.anchors)


def test_trace_batch_below_two_aborts(exporter):
    with pytest.raises(SystemExit, match="trace-batch"):
        exporter.export(argparse.Namespace(trace_batch=1))


def test_dynamic_batch_check_requires_onnxruntime(exporter, monkeypatch):
    real_import = builtins.__import__

    def without_onnxruntime(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("simulated missing onnxruntime")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_onnxruntime)
    with pytest.raises(SystemExit, match="requires numpy and onnxruntime"):
        exporter._verify_dynamic_batch_runs(Path("unused.onnx"), {})
    with pytest.raises(SystemExit, match="requires numpy and onnxruntime"):
        exporter.export(
            argparse.Namespace(
                trace_batch=2,
                output="model.onnx",
                checkpoint="model.pt",
            )
        )
