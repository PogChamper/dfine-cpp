# Superseded experiment scripts
One-offs from the M0 deformable-attention investigation, kept for provenance. Their logic is now in
`../export_dfine_onnx.py` (the `--deform {explicit,gridsample}` flag + the explicit gather-bilinear core).
- `export_dynamo_op19.py` — early opset-19/dynamo raw export probe.
- `spike_explicit_deform.py` — the original explicit-deform validation spike.
See `../../../docs/impl/M0_STATUS.md` for the full story.
