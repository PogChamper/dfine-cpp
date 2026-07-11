# Validation

The reference accuracy and research campaign ran on an RTX 4070 Ti SUPER (Ada, SM 8.9), WSL2, and TensorRT 10.13. The release `slim` recipe was then rebuilt and measured on RTX 3090 (Ampere) and RTX 5080 (Blackwell) systems. The three generations reproduce its accuracy and the expected recipe ordering.

TensorRT engines are compiled on the target stack. `validation_report.py` records environment facts, artifact hashes, the exact engine recipe, and steady-state batch-1/batch-8 throughput in a comparable report. The published rows below are maintainer-run; external reports are welcome and identified separately when submitted.

The v0.3.1, v0.3.2, and v0.3.3 model artifacts are byte-identical. Rows produced with v0.3.2 therefore validate the current graph recipe; the `dfine` column records the tooling version used for each report.

## Generate a compatibility report

Requirements: a repository checkout and Python ≥3.10. The engine build also needs `tensorrt-cu12==10.13.*`. Without TensorRT, the report records the environment and marks the build skipped. If TensorRT is present but no GPU is usable, the build is recorded as failed.

```sh
git clone https://github.com/PogChamper/dfine-cpp && cd dfine-cpp
python -m pip install "tensorrt-cu12==10.13.*"
curl -fLO https://github.com/PogChamper/dfine-cpp/releases/download/v0.3.3/dfine_m_slim.onnx \
     -fLO https://github.com/PogChamper/dfine-cpp/releases/download/v0.3.3/dfine_m_slim.json \
     -fLO https://github.com/PogChamper/dfine-cpp/releases/download/v0.3.3/SHA256SUMS
python trt-files/scripts/validation_report.py --onnx dfine_m_slim.onnx \
    --check-sums SHA256SUMS --out validation
```

The report uses the production engine recipe: strong typing, TF32 disabled, and max batch 8. Build `dfine_bench` with `./build.sh` before running the report to include throughput; otherwise the engine and environment sections remain useful.

## Run full COCO accuracy

Build `dfine_coco_eval`, then point the scorer at COCO val2017:

```sh
: "${COCO_IMAGES:?set COCO_IMAGES to COCO val2017}"
: "${COCO_ANN:?set COCO_ANN to instances_val2017.json}"
: "${TRTLIB:?set TRTLIB to the directory containing libnvinfer.so.10}"
mkdir -p validation
uv run python trt-files/scripts/cpp_coco_eval.py \
    --binary build/dfine_coco_eval \
    --engine dfine_m_slim.engine \
    --images "$COCO_IMAGES" \
    --ann "$COCO_ANN" \
    --ld-library-path "$TRTLIB" \
    --batch 8 --limit 0 --out validation/coco_detections.json
```

## v0.4.0 release-candidate gate

The unreleased runtime was revalidated on 2026-07-11 on the reference RTX 4070 Ti SUPER stack.
Full COCO val2017 used the same engine bytes as the recorded v0.3.3 reference, batch 8, and the
complete C++ preprocessing/decode path. All seven artifacts matched v0.3.3 at three-decimal report
precision.

| Artifact | Queries | AP |
|---|---:|---:|
| `slim` N | 300 | 0.428 |
| `slim` S | 300 | 0.506 |
| `slim` M | 300 | 0.550 |
| `slim` L | 300 | 0.572 |
| `slim` X | 300 | 0.593 |
| reduced-query N | 100 | 0.423 |
| reduced-query M | 100 | 0.545 |

The reduced-query rows use the fast graph: Q200 followed by cascade `1:100`, producing Q=100
outputs. Interleaved v0.3.3/RC checks kept all five `slim` engines within normal run-to-run variation
and found unchanged TensorRT inference time for the reduced-query engines.

## Results

| GPU | SM | TRT | Driver | OS | dfine | build ok | b1 img/s | b8 img/s | submitted-by |
|-----|----|-----|--------|----|-------|----------|----------|----------|--------------|
| RTX 4070 Ti SUPER | 8.9 | 10.13.3.9 | 581.15 | WSL2 (6.18) | 0.3.2 | yes (132.8 s) | 279.5 | 506.1 | maintainer |
| RTX 3090 | 8.6 | 10.13.3.9 | 550.107.02 | Ubuntu 22.04 native | 0.3.2 | yes | 310 | 487 | maintainer (rented) |
| RTX 5080 | 12.0 | 10.13.3.9 | 610.43.02 | Ubuntu 22.04 native | 0.3.2 | yes | 456 | 676 | maintainer (rented) |

## Cross-GPU benchmark matrix

The rows above are the D-FINE-M `slim` spot bench from `validation_report.py`. The tables below
reproduce the README benchmark methodology end to end on rented hardware — batches 1/2/4/8
× 500 iters, medians of 3 interleaved rounds, peak VRAM — via
`trt-files/scripts/remote_matrix.sh` followed by
`trt-files/scripts/remote_bench_full.sh`. The mAP column is
the 500-image val2017 subset: a lossless check against fp32 on the same machine, **not**
comparable to the README's full-val numbers. Cells are `p50 ms / img/s`.

The fast/max rows report the measured `K=Q` reduced-query configuration. The Ada release-candidate
gate above reproduced N/M accuracy and TensorRT inference timing.

### RTX 3090 — Ampere SM 8.6, driver 550.107.02, TRT 10.13.3.9, Ubuntu 22.04 native

| size | config | b1 | b2 | b4 | b8 | VRAM MiB | subset mAP |
|------|--------|----|----|----|----|----------|------------|
| m | legacy `fp16_st` | 3.55/281 | 6.13/326 | 10.58/378 | 19.23/416 | 532 | — |
| m | `slim` | 3.23/310 | 5.39/371 | 9.11/439 | 16.44/487 | 468 | 0.578 |
| m | fast | 3.07/326 | 4.99/401 | 8.12/493 | 14.25/562 | 458 | 0.571 |
| m | max | 3.90/257 | 5.33/375 | 8.13/492 | 13.67/585 | 428 | 0.571 |
| m | fp32 | 5.95/168 | 10.64/188 | 19.80/202 | 38.38/208 | 650 | 0.577 |
| n | legacy `fp16_st` | 1.66/601 | 2.49/802 | 4.05/988 | 7.12/1124 | 232 | — |
| n | `slim` | 1.51/664 | 2.25/887 | 3.65/1098 | 6.33/1265 | 204 | 0.458 |
| n | fast | 1.38/725 | 1.97/1014 | 3.05/1310 | 5.17/1549 | 202 | 0.455 |
| n | fp32 | 2.27/440 | 3.50/571 | 5.97/670 | 10.67/750 | 266 | 0.458 |

In those runs, the batch-8 ordering was `legacy fp16_st < slim < fast < max`;
`slim` matched FP32 on the subset, and the fast/max deltas matched the Ada reference. Native
Linux also reduced the batch-1 dispatch cost measured under WSL2.

### RTX 5080 — Blackwell SM 12.0, driver 610.43.02, TRT 10.13.3.9, Ubuntu 22.04 native

| size | config | b1 | b2 | b4 | b8 | VRAM MiB | subset mAP |
|------|--------|----|----|----|----|----------|------------|
| m | legacy `fp16_st` | 2.29/436 | 3.73/536 | 6.42/623 | 12.44/643 | 542 | — |
| m | `slim` | 2.19/456 | 3.55/564 | 6.10/656 | 11.83/676 | 476 | 0.575 |
| m | fast | 1.84/542 | 2.88/696 | 4.80/834 | 9.19/871 | 468 | 0.571 |
| m | max | 2.21/452 | 3.03/661 | 4.59/872 | 8.08/990 | 440 | 0.570 |
| m | fp32 | 3.93/254 | 6.53/306 | 11.98/334 | 24.78/323 | 702 | 0.577 |
| n | legacy `fp16_st` | 1.00/999 | 1.61/1245 | 2.53/1579 | 4.39/1821 | 236 | — |
| n | `slim` | 0.91/1099 | 1.39/1439 | 2.34/1711 | 4.10/1953 | 208 | 0.457 |
| n | fast | 0.82/1222 | 1.18/1702 | 1.90/2111 | 3.19/2506 | 206 | 0.454 |
| n | fp32 | 1.50/665 | 2.22/900 | 3.55/1126 | 6.86/1166 | 268 | 0.458 |

The `sm_89` wheel runs on RTX 5080 through PTX JIT with matching detections. On this system the 0-aux-stream `slim-g0` build also gains throughput at batch 8; the Ada and Ampere runs were batch-8-neutral.

## Submit a report

Inspect `validation/report.md` and `validation/report.json` (`schema: 1`), then attach both to a
GitHub issue titled `validation: <GPU> / TRT <version>`. The report includes `nvidia-smi` and
`platform.uname()` output; review it before posting.
