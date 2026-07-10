#!/usr/bin/env python3
"""Insert INT8 QDQ nodes into a raw D-FINE ONNX over the backbone+encoder only.

Explicit (QDQ) quantization, NOT TensorRT's deprecated implicit IInt8EntropyCalibrator2:
onnxruntime.quantization.quantize_static places QuantizeLinear/DequantizeLinear pairs
around Conv/MatMul, with scales calibrated on real COCO images. The decoder is excluded
by name (all its nodes are cleanly scoped under /model/decoder + model.decoder), so it
carries no Q/DQ and TensorRT runs it in FP32 — the FP-sensitive FDR path stays faithful,
exactly as in the FP16 mixed build. Build the result with `build_engine.py --int8`.

    convert_int8.py --onnx dfine_m.onnx --output dfine_m_int8_qdq.onnx --num-calib 200

Expect some mAP loss vs FP32/FP16 (INT8 on a detection transformer is lossy); quantify
it with profile.py before trusting the engine.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import onnx

sys.path[:] = [p for p in sys.path if p not in ("", str(Path(__file__).resolve().parent))]
sys.path.append(str(Path(__file__).resolve().parent))
import cv2  # noqa: E402
from coco_eval import preprocess  # noqa: E402  (the exact D-FINE /255 stretch pipeline)

from onnxruntime.quantization import (  # noqa: E402
    CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process  # noqa: E402



def _publish_pair(graph_tmp, graph_out, sidecar_text, sidecar_out, tag):
    """Publish a staged graph and its (optional) sidecar with the smallest
    possible inconsistency window: the sidecar is staged BEFORE either swap,
    then both land via two adjacent atomic renames (each rename atomic; the
    pair is not jointly transactional — the window is two syscalls).
    sidecar_text=None means this producer has no contract to carry through;
    a sidecar already sitting at sidecar_out would then describe the PREVIOUS
    graph, so it is removed in the same publish step."""
    graph_tmp, graph_out = Path(graph_tmp), Path(graph_out)
    sidecar_out = Path(sidecar_out)
    sc_tmp = None
    if sidecar_text is not None:
        sc_tmp = Path(str(sidecar_out) + ".tmp")
        sc_tmp.write_text(sidecar_text)
    os.replace(graph_tmp, graph_out)
    if sc_tmp is not None:
        os.replace(sc_tmp, sidecar_out)
    elif sidecar_out.exists():
        sidecar_out.unlink()
        print(f"[{tag}] removed stale sidecar {sidecar_out} (source has none)")

def decoder_node_names(model: onnx.ModelProto, prefixes: tuple[str, ...]) -> list[str]:
    """Every node whose name is under the decoder scope — these stay FP32 (no Q/DQ)."""
    return [n.name for n in model.graph.node if n.name.startswith(prefixes)]


class CocoCalib(CalibrationDataReader):
    """Feeds preprocessed COCO images as the 'images' input for scale calibration."""

    def __init__(self, images_dir: Path, file_names: list[str], img_size: int, input_name: str):
        self.images_dir = images_dir
        self.file_names = file_names
        self.img_size = img_size
        self.input_name = input_name
        self._it = iter(file_names)

    def get_next(self):
        for fname in self._it:
            bgr = cv2.imread(str(self.images_dir / fname), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            x = preprocess(bgr, self.img_size).numpy().astype(np.float32)  # [1,3,H,W]
            return {self.input_name: x}
        return None

    def rewind(self):
        self._it = iter(self.file_names)


def main(args: argparse.Namespace) -> None:
    onnx_path = Path(args.onnx).resolve()
    out_path = Path(args.output).resolve()
    model = onnx.load(str(onnx_path))
    input_name = model.graph.input[0].name

    prefixes = tuple(p for p in args.decoder_prefixes.split(",") if p)
    exclude = decoder_node_names(model, prefixes)
    print(f"[int8] excluding {len(exclude)} decoder nodes from quantization (stay FP32)")

    # Pick calibration images (deterministic first-N of the sorted annotation).
    from pycocotools.coco import COCO
    coco = COCO(args.ann)
    img_ids = sorted(coco.getImgIds())[: args.num_calib]
    file_names = [coco.loadImgs(i)[0]["file_name"] for i in img_ids]
    print(f"[int8] calibrating on {len(file_names)} images from {args.images}")

    # ORT wants a shape-inferred, cleaned model before static quantization.
    pre_path = out_path.with_suffix(".preproc.onnx")
    quant_pre_process(str(onnx_path), str(pre_path), skip_symbolic_shape=True)

    reader = CocoCalib(Path(args.images), file_names, args.img_size, input_name)
    method = {"minmax": CalibrationMethod.MinMax,
              "entropy": CalibrationMethod.Entropy,
              "percentile": CalibrationMethod.Percentile}[args.calib_method]

    quant_tmp = Path(str(out_path) + ".tmp")
    quantize_static(
        str(pre_path), str(quant_tmp), reader,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=["Conv", "MatMul"],  # the compute-heavy backbone/encoder ops
        nodes_to_exclude=exclude,                  # decoder stays FP32
        per_channel=True,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=method,
        # TensorRT INT8 QDQ constraints: SYMMETRIC only (zero-point 0; ORT defaults to
        # asymmetric activations), and NO bias quantization (TRT folds bias into the
        # INT8 conv itself; a DequantizeLinear on the int32 bias is rejected).
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True,
                       "QuantizeBias": False},
    )
    pre_path.unlink(missing_ok=True)

    q = onnx.load(str(quant_tmp))
    n_q = sum(1 for n in q.graph.node if n.op_type == "QuantizeLinear")
    n_dq = sum(1 for n in q.graph.node if n.op_type == "DequantizeLinear")

    # Carry the descriptive sidecar through so the C++ runtime stays model-generic.
    side = onnx_path.with_suffix(".json")
    sidecar_text = None
    if side.exists():
        import json
        meta = json.loads(side.read_text())
        meta["precision"] = "int8"
        meta["quant"] = "qdq_backbone_encoder"
        sidecar_text = json.dumps(meta, indent=2) + "\n"
    _publish_pair(quant_tmp, out_path, sidecar_text, out_path.with_suffix(".json"), "int8")
    print(f"[int8] wrote {out_path}: {n_q} QuantizeLinear / {n_dq} DequantizeLinear nodes")
    if sidecar_text is not None:
        print(f"[int8] wrote sidecar {out_path.with_suffix('.json')}")


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Insert INT8 QDQ into a D-FINE ONNX (backbone/encoder only)")
    p.add_argument("--onnx", default=str(repo / "trt-files" / "onnx" / "dfine_m.onnx"))
    p.add_argument("--output", default=str(repo / "trt-files" / "onnx" / "dfine_m_int8_qdq.onnx"))
    p.add_argument("--images", default="/mnt/d/datasets/coco/val2017")
    p.add_argument("--ann", default="/mnt/d/datasets/coco/annotations/instances_val2017.json")
    p.add_argument("--num-calib", type=int, default=200)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--calib-method", choices=["minmax", "entropy", "percentile"], default="minmax")
    p.add_argument("--decoder-prefixes", default="/model/decoder,model.decoder")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
