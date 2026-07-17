from __future__ import annotations

import gc
import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO / "trt-files" / "dfine_model"
sys.path.insert(0, str(MODEL_ROOT.parent))

from dfine_model import build_model  # noqa: E402, I001


TOPOLOGY = {
    "n": (674, "7cc0efd8f91995433ed300b048b43ac3535a6d0f02c8e3c13c44afc84a83714f"),
    "s": (794, "88b6ca9acb19b4e78c9708b4d418f4a297bb3e4d9d44a057caf74133751f6547"),
    "m": (1053, "581e6cc565657c23dc85509c8eab0580ff13387dcccf276e84367678bbda8173"),
    "l": (1173, "e906ac31d22c930c8bb64e85d51c9ae7de09fd7cb552628cdad22b45b4ef1c4e"),
    "x": (1441, "15ba8940994f447b7118d040ef75f0fe021fec30009c2ff49d5b70e3d20465ec"),
}

CHECKPOINTS = {
    "n": "dfine_n_coco.pt",
    "s": "dfine_s_obj2coco.pt",
    "m": "dfine_m_obj2coco.pt",
    "l": "dfine_l_obj2coco.pt",
    "x": "dfine_x_obj2coco.pt",
}


def topology_digest(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(b"\0")
    return digest.hexdigest()


@pytest.mark.parametrize("variant", TOPOLOGY)
def test_variant_topology(variant):
    expected_count, expected_digest = TOPOLOGY[variant]
    model = build_model(variant, 80, (640, 640))
    assert len(model.state_dict()) == expected_count
    assert topology_digest(model) == expected_digest


def test_nano_deploy_forward_contract():
    model = build_model("n", 7, (640, 640)).deploy()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 640, 640))
    assert output["pred_logits"].shape == (1, 300, 7)
    assert output["pred_boxes"].shape == (1, 300, 4)


def test_cascade_forward_contract():
    path = REPO / "trt-files" / "scripts" / "export_dfine_onnx.py"
    spec = importlib.util.spec_from_file_location("bundled_model_exporter", path)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)

    model = build_model("n", 7, (640, 640))
    exporter.apply_sliders(
        model,
        SimpleNamespace(eval_idx=None, num_queries=200, cascade="1:100"),
    )
    model.deploy()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 640, 640))
    assert output["pred_logits"].shape == (1, 100, 7)
    assert output["pred_boxes"].shape == (1, 100, 4)


def test_cascade_score_head_follows_model_device():
    path = REPO / "trt-files" / "scripts" / "export_dfine_onnx.py"
    spec = importlib.util.spec_from_file_location("bundled_model_exporter_device", path)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)

    model = build_model("n", 7, (640, 640))
    exporter.apply_sliders(
        model,
        SimpleNamespace(eval_idx=None, num_queries=200, cascade="1:100"),
    )
    model.to("meta")

    head = model.decoder.decoder.cascade_score_head
    assert next(head.parameters()).device.type == "meta"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cascade_cuda_forward_contract():
    path = REPO / "trt-files" / "scripts" / "export_dfine_onnx.py"
    spec = importlib.util.spec_from_file_location("bundled_model_exporter_cuda", path)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)

    model = build_model("n", 7, (640, 640)).cuda()
    exporter.apply_sliders(
        model,
        SimpleNamespace(eval_idx=None, num_queries=200, cascade="1:100"),
    )
    model.deploy()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 640, 640, device="cuda"))
    assert output["pred_logits"].shape == (1, 100, 7)
    assert output["pred_boxes"].shape == (1, 100, 4)


def test_pinned_dfine_seg_oracle():
    source = os.environ.get("DFINE_SEG_SRC") or os.environ.get("DFINE_SEG_DIR")
    checkpoint_dir = os.environ.get("DFINE_CHECKPOINT_DIR")
    if not source or not checkpoint_dir:
        pytest.skip("set DFINE_SEG_SRC and DFINE_CHECKPOINT_DIR for differential parity")

    source_path = Path(source).resolve()
    revision = (REPO / "trt-files" / "DFINE_SEG_REVISION").read_text().strip()
    import subprocess

    commit = subprocess.run(
        ["git", "-C", str(source_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert commit == revision

    sys.path.insert(0, str(source_path))
    from src.d_fine.dfine import build_model as build_oracle

    image = torch.rand((1, 3, 640, 640), generator=torch.Generator().manual_seed(7))
    for variant, filename in CHECKPOINTS.items():
        checkpoint = Path(checkpoint_dir) / filename
        if not checkpoint.exists():
            checkpoint = Path(checkpoint_dir) / "pretrained" / filename
        assert checkpoint.is_file(), checkpoint
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        local = build_model(variant, 80, (640, 640))
        oracle = build_oracle(
            variant,
            80,
            False,
            "cpu",
            img_size=(640, 640),
            pretrained_model_path=None,
            pretrained_backbone=False,
        )
        local.load_state_dict(state, strict=True)
        oracle.load_state_dict(state, strict=True)
        for deployed in (False, True):
            if deployed:
                local.deploy()
                oracle.deploy()
            else:
                local.eval()
                oracle.eval()
            with torch.inference_mode():
                actual = local(image)
                expected = oracle(image)
            assert torch.equal(actual["pred_logits"], expected["pred_logits"])
            assert torch.equal(actual["pred_boxes"], expected["pred_boxes"])
        del local, oracle, state
        gc.collect()
