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
`pip install "tensorrt-cu12==10.13.*"` — without it (or without a GPU) the script still
writes a useful report with the build marked "skipped".

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
| RTX 5080 | 12.0 | 10.13.3.9 | 610.43.02 | Ubuntu 22.04 native | 0.3.2 | yes | 456 | 676 | maintainer (rented) |

## Full-methodology results (maintainer-run)

The rows above are the m-surgical spot bench from `validation_report.py`. The tables below
reproduce the README benchmark methodology end to end on rented hardware — batches 1/2/4/8
× 500 iters, medians of 3 interleaved rounds, peak VRAM — via
`trt-files/scripts/remote_matrix.sh` followed by `remote_bench_full.sh`. The mAP column is
the 500-image val2017 subset: a lossless check against fp32 on the same machine, **not**
comparable to the README's full-val numbers. Cells are `p50 ms / img/s`.

### RTX 3090 — Ampere SM 8.6, driver 550.107.02, TRT 10.13.3.9, Ubuntu 22.04 native

| size | config | b1 | b2 | b4 | b8 | VRAM MiB | subset mAP |
|------|--------|----|----|----|----|----------|------------|
| m | prod | 3.55/281 | 6.13/326 | 10.58/378 | 19.23/416 | 532 | — |
| m | surgical | 3.23/310 | 5.39/371 | 9.11/439 | 16.44/487 | 468 | 0.578 |
| m | fast | 3.07/326 | 4.99/401 | 8.12/493 | 14.25/562 | 458 | 0.571 |
| m | max | 3.90/257 | 5.33/375 | 8.13/492 | 13.67/585 | 428 | 0.571 |
| m | fp32 | 5.95/168 | 10.64/188 | 19.80/202 | 38.38/208 | 650 | 0.577 |
| n | prod | 1.66/601 | 2.49/802 | 4.05/988 | 7.12/1124 | 232 | — |
| n | surgical | 1.51/664 | 2.25/887 | 3.65/1098 | 6.33/1265 | 204 | 0.458 |
| n | fast | 1.38/725 | 1.97/1014 | 3.05/1310 | 5.17/1549 | 202 | 0.455 |
| n | fp32 | 2.27/440 | 3.50/571 | 5.97/670 | 10.67/750 | 266 | 0.458 |

The full tier ladder reproduces (prod < surgical < fast < max), surgical is lossless
against fp32 on the subset, and the fast/max accuracy deltas match Ada's. Batch-1 latency
beats the published WSL2 numbers across the board — D-FINE is dispatch-bound at small
batch, and native Linux does not pay the WSL2 kernel-launch tax the README table carries.

### RTX 5080 — Blackwell SM 12.0, driver 610.43.02, TRT 10.13.3.9, Ubuntu 22.04 native

| size | config | b1 | b2 | b4 | b8 | VRAM MiB | subset mAP |
|------|--------|----|----|----|----|----------|------------|
| m | prod | 2.29/436 | 3.73/536 | 6.42/623 | 12.44/643 | 542 | — |
| m | surgical | 2.19/456 | 3.55/564 | 6.10/656 | 11.83/676 | 476 | 0.575 |
| m | fast | 1.84/542 | 2.88/696 | 4.80/834 | 9.19/871 | 468 | 0.571 |
| m | max | 2.21/452 | 3.03/661 | 4.59/872 | 8.08/990 | 440 | 0.570 |
| m | fp32 | 3.93/254 | 6.53/306 | 11.98/334 | 24.78/323 | 702 | 0.577 |
| n | prod | 1.00/999 | 1.61/1245 | 2.53/1579 | 4.39/1821 | 236 | — |
| n | surgical | 0.91/1099 | 1.39/1439 | 2.34/1711 | 4.10/1953 | 208 | 0.457 |
| n | fast | 0.82/1222 | 1.18/1702 | 1.90/2111 | 3.19/2506 | 206 | 0.454 |
| n | fp32 | 1.50/665 | 2.22/900 | 3.55/1126 | 6.86/1166 | 268 | 0.458 |

First Blackwell build of this project (`CUDA_ARCH=120`, driver 610-open). Everything
holds: tier ladder, lossless surgical, exact predict parity on the README picture. The
published sm_89 wheel runs via PTX JIT with identical detections — RTX 50xx works out of
the box. One scheduler difference: the 0-aux-stream CUDA-graph build (`slim-g0`) gains
throughput even at batch 8 here, where Ada and Ampere are batch-8-neutral.
