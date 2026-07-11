#!/usr/bin/env python3
"""Smoke-test a D-FINE TensorRT engine across the dynamic batch range.

For each requested batch size it sets the input shape, binds torch CUDA tensors as
the I/O buffers, runs one ``execute_async_v3``, and prints the resolved output
shapes/dtypes. This is the Python analogue of the C++ ``dfine_smoke`` app: the
engine must bind and run at N=1 and N=max.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorrt as trt
import torch
from eval_contract import positive_int

_TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
    trt.DataType.BOOL: torch.bool,
}


def load_engine(path: Path, logger: trt.Logger) -> trt.ICudaEngine:
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise RuntimeError(f"failed to deserialize {path}")
    return engine


def run_batch(
    engine: trt.ICudaEngine,
    context: trt.IExecutionContext,
    names,
    n: int,
    img: int,
    stream: torch.cuda.Stream,
) -> None:
    buffers = {}
    input_names = [name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]
    if len(input_names) != 1:
        raise RuntimeError(f"expected one input tensor, found {len(input_names)}")
    for name in names:
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            if not context.set_input_shape(name, (n, 3, img, img)):
                raise RuntimeError(f"set_input_shape failed for {name} at N={n}")

    for name in names:
        shape = tuple(context.get_tensor_shape(name))
        if not shape or any(dim <= 0 for dim in shape):
            raise RuntimeError(f"unresolved shape for {name} at N={n}: {shape}")
        if shape[0] != n:
            raise RuntimeError(f"{name} resolved to batch {shape[0]} at requested N={n}")
        trt_dtype = engine.get_tensor_dtype(name)
        if trt_dtype not in _TRT_TO_TORCH:
            raise RuntimeError(f"unsupported dtype for {name}: {trt_dtype}")
        dtype = _TRT_TO_TORCH[trt_dtype]
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            buf = torch.rand(shape, dtype=dtype, device="cuda")
        else:
            buf = torch.empty(shape, dtype=dtype, device="cuda")
        buffers[name] = buf
        if not context.set_tensor_address(name, buf.data_ptr()):
            raise RuntimeError(f"set_tensor_address failed for {name} at N={n}")

    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError(f"execute_async_v3 failed at N={n}")
    stream.synchronize()

    desc = "  ".join(
        f"{name}{tuple(buffers[name].shape)}:{str(buffers[name].dtype).split('.')[-1]}"
        for name in names
    )
    print(f"[smoke] N={n:<3} OK   {desc}")


def main(args: argparse.Namespace) -> int:
    engine_path = Path(args.engine).resolve()
    logger = trt.Logger(trt.Logger.WARNING)
    engine = load_engine(engine_path, logger)
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("failed to create TensorRT execution context")
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    modes = {
        name: ("in" if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else "out")
        for name in names
    }
    print(f"[smoke] engine={engine_path.name} TensorRT {trt.__version__}")
    tensors = ", ".join(f"{name}({modes[name]},{engine.get_tensor_dtype(name)})" for name in names)
    print("[smoke] tensors: " + tensors)

    stream = torch.cuda.Stream()
    for n in args.batches:
        run_batch(engine, context, names, n, args.img_size, stream)
    print("[smoke] all batch sizes bound and ran")
    return 0


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Smoke-test a D-FINE TensorRT engine")
    p.add_argument("--engine", default=str(repo / "trt-files" / "engines" / "dfine_m_fp32.engine"))
    p.add_argument("--img-size", type=positive_int, default=640)
    p.add_argument("--batches", type=positive_int, nargs="+", default=[1, 8])
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
