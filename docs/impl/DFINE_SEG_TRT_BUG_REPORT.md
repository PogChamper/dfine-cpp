# D-FINE on TensorRT loses ~10–18 AP — root cause + one-function fix

**TL;DR.** The exported D-FINE model loses a large amount of mAP **only on TensorRT** (PyTorch and ONNXRuntime are unaffected). The cause is TensorRT's in-context compilation of the **`F.grid_sample`-based deformable-attention core**, whose tiny error is amplified into box drift by D-FINE's FDR distribution decode. Replacing `grid_sample` with an explicit **gather-bilinear** of the identical math (no plugin) recovers full mAP at no latency cost. `run_parity` does not catch it because it compares **scores only**, and the drift is in **boxes**.

Environment: RTX 4070 Ti SUPER (Ada), CUDA 12.8, **TensorRT 10.13.3**, torch 2.9.1, onnxruntime 1.24.4. Model: `dfine_m_obj2coco.pt`, COCO val2017.

## Evidence (COCO val2017)

Same images, same decode where comparable:

| Backend / export | AP@[.50:.95] | notes |
|---|---|---|
| PyTorch (reference) | **0.5509** (full-val) / 0.5672 (2000-subset) | matches the paper |
| ONNXRuntime (same ONNX) | = PyTorch to 4 decimals | the graph is correct |
| **TensorRT, stock `export.py`** (opset19/dynamo/onnxsim, fused postproc) | **0.3621** (2000) | **their recommended path — broken** |
| **TensorRT, same `export.py` + the fix** | **0.5582** (2000) | **their exact path, recovered (+19.6 AP)** |
| TensorRT, raw export, grid_sample (opset16) | 0.4455 (full-val) / 0.4572 (2000) | clean isolation |
| TensorRT, raw export, explicit gather-bilinear | **0.5507** (full-val) / 0.5669 (2000) | = PyTorch |

Latency (RTX 4070 Ti SUPER, N=1): grid_sample 5.04 ms vs explicit **4.92 ms** — the fix is *not* slower.

Validated across all sizes (2000-subset, ORT-GPU exact reference vs explicit TRT engine, ≤0.0001 AP each): n 0.444/0.444, s 0.524/0.524, l 0.592/0.592, x 0.607/0.607.

## Root cause (localized)

TRT's GridSample is **bit-exact in isolation** (single-node test, incl. out-of-bounds: max abs 2.4e-7). The drift appears only **in context**. Forward-hook bisection (TRT vs ORT, surviving queries):

- Divergence enters **suddenly at decoder layer 1** (L0 output exact 1.9e-6 → L1 0.19).
- Within layer 1: self-attn exact, LayerNorm exact, **deformable cross-attn diverges (7e-2)**, Gate fusion amplifies.
- The FDR `bbox_head → Integral → distance2bbox` path is itself bit-exact; layer-0 (same op, exact inputs) is exact — so TRT picks a **different, less-accurate kernel for the deformable-attn matmul/weighted-sum at layers 1–3**, and the FDR cross-layer box accumulation bakes it in. Scores (single linear head, no accumulation) stay faithful → `run_parity` (sorted-topK **scores** only) passes while boxes drift.

No builder lever fixes it: TF32-off, `PREFER_PRECISION_CONSTRAINTS`, `builder_optimization_level=5`, strongly-typed, `set_tactic_sources`, opset16-legacy vs opset19-dynamo — all unchanged.

## The fix

In `src/d_fine/arch/utils.py`, replace the `F.grid_sample` call in `deformable_attention_core_func_v2` (the `method == "default"` branch) with an explicit gather-bilinear that reproduces `grid_sample(mode="bilinear", padding_mode="zeros", align_corners=False)` exactly. The rest of the function (offset/weight projections, weighted sum) is unchanged. Torch parity of the rewrite: logits max-abs 2.3e-4.

```python
def _bilinear_gather(value_l, grid_l, h, w):
    # value_l [M,C,h,w], grid_l [M,Lq,P,2] in [-1,1]  ->  [M,C,Lq,P]
    M, c = value_l.shape[0], value_l.shape[1]
    Lq, P = grid_l.shape[1], grid_l.shape[2]
    gx, gy = grid_l[..., 0], grid_l[..., 1]
    ix = (gx + 1) * w / 2 - 0.5            # align_corners=False
    iy = (gy + 1) * h / 2 - 0.5
    x0, y0 = torch.floor(ix), torch.floor(iy)
    x1, y1 = x0 + 1, y0 + 1
    wx1, wy1 = ix - x0, iy - y0
    wx0, wy0 = 1 - wx1, 1 - wy1
    vflat = value_l.reshape(M, c, h * w)
    def clip(t, hi):  # NOT .clamp(): dynamo lowers .clamp() to a Clip whose const inputs TRT 10.13
        return torch.minimum(torch.maximum(t, t.new_zeros(())), t.new_full((), float(hi)))  # rejects
    def corner(xc, yc, wgt):
        valid = ((xc >= 0) & (xc <= w - 1) & (yc >= 0) & (yc <= h - 1)).to(value_l.dtype)
        idx = (clip(yc, h - 1) * w + clip(xc, w - 1)).long().reshape(M, 1, Lq * P).expand(M, c, Lq * P)
        return torch.gather(vflat, 2, idx).reshape(M, c, Lq, P) * (wgt * valid).unsqueeze(1)
    return (corner(x0, y0, wx0 * wy0) + corner(x1, y0, wx1 * wy0)
            + corner(x0, y1, wx0 * wy1) + corner(x1, y1, wx1 * wy1))

# inside deformable_attention_core_func_v2, default branch:
#   sampling_value_l = _bilinear_gather(value_l, sampling_grid_l, int(h), int(w))
```

This removes every `GridSample` node from the ONNX (replaced by `Gather` + arithmetic), which TRT executes in exact FP32.

## Notes for integrating into `export.py`

- **Drop-in for the current default export** (`dynamo=True`, opset 19, `onnxsim`, fused postproc): measured end-to-end via D-FINE-seg's own `export_to_onnx`/`export_to_tensorrt`, **0.3621 → 0.5582** (their exact path). The only subtlety is the index clamp: write it as `minimum(maximum(...))`, **not** `.clamp()` — the dynamo exporter lowers `.clamp()` to a `Clip` whose constant min/max inputs TensorRT 10.13's parser rejects ("input was not registered"), which also makes `onnxsim` emit a dangling `Clip`. With `min/max`, both `dynamo`/opset-19 and the legacy opset-16 tracer parse + build cleanly, and onnxsim is happy.
- Consider extending `run_parity` to also compare **box geometry** (or per-backend mAP), since the scores-only check is blind to this class of bug.

## Repro

`trt-files/scripts/seg_export_repro.py` (uses D-FINE-seg's own `export_to_onnx`/`export_to_tensorrt` + `DFINEPostProcessor`/`ExportWrapper`), and `export_dfine_onnx.py --deform {gridsample,explicit}` for the clean isolation.
