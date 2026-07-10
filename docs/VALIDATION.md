# External validation matrix

Every number this repo publishes — parity, mAP, throughput — was measured on **one machine**:
an RTX 4070 Ti SUPER (Ada, SM 8.9), WSL2, TensorRT 10.x. The engine, however, is compiled on
*your* GPU from the released ONNX, so behavior on other architectures (Ampere, Hopper, Orin, …),
drivers and OSes is asserted, not demonstrated. This page closes that gap: a stranger with a
different GPU runs one script and gets a report comparable across machines, because the tool
records the same facts the same way — environment, artifact hashes, the exact build recipe,
and steady-state throughput.

## Run it

Requirements: a repo checkout and Python ≥ 3.10. For the engine-build step also
`pip install "tensorrt==10.*"` — without it (or without a GPU) the script still writes a
useful report with the build marked "skipped".

```sh
git clone https://github.com/PogChamper/dfine-cpp && cd dfine-cpp
curl -LO https://github.com/PogChamper/dfine-cpp/releases/download/v0.3.2/dfine_m_slim.onnx \
     -LO https://github.com/PogChamper/dfine-cpp/releases/download/v0.3.2/dfine_m_slim.json \
     -LO https://github.com/PogChamper/dfine-cpp/releases/download/v0.3.2/SHA256SUMS
python trt-files/scripts/validation_report.py --onnx dfine_m_slim.onnx \
    --check-sums SHA256SUMS --out validation
```

The engine build is the README quickstart recipe (`build_engine.py --no-tf32 --max-batch 8`,
`--strongly-typed` added automatically for fp16-typed ONNX per the sidecar) and takes 1–3 min.
Throughput rows appear only if the C++ bench tool exists: run `./build.sh` first, then rerun the
script — it picks up `build/dfine_bench` automatically and benches batches 1 and 8.

## Submit

Attach **both** files from `validation/` — `report.md` (human-readable) and `report.json`
(machine-readable, `schema: 1`) — to a GitHub issue titled:

    validation: <GPU> / TRT <version>

e.g. `validation: RTX 3060 / TRT 10.9`. Nothing in the report identifies you beyond what
`nvidia-smi` and `platform.uname()` print; skim it before posting if in doubt.

## Results

| GPU | SM | TRT | Driver | OS | dfine | build ok | b1 img/s | b8 img/s | submitted-by |
|-----|----|-----|--------|----|-------|----------|----------|----------|--------------|
| RTX 4070 Ti SUPER | 8.9 | 10.13.3.9 | 581.15 | WSL2 (6.18) | 0.3.2 | yes (132.8 s) | 279.5 | 506.1 | maintainer |
| RTX 3090 | 8.6 | 10.13.3.9 | 550.107.02 | Ubuntu 22.04 native | 0.3.2 | yes | 310 | 487 | maintainer (rented) |
