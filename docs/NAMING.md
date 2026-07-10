# Artifact naming & identity

Rule of thumb: **the name is a label, the sidecar is the truth, the fingerprint
is the identity.** Tools may use filenames to *find* candidates, but any fact
that decides behavior (batch profile, precision recipe, source binding) must be
read from the artifact's JSON sidecar, never parsed out of its name.

## The four identity axes

| # | Axis | Question it answers | Recorded in | Typical fields |
|---|------|--------------------|-------------|----------------|
| 1 | Model contract | *what* is computed | ONNX sidecar | `variant`, `checkpoint_sha256`, `num_classes`, `class_names`, input size, preprocessing, output contract |
| 2 | Conversion recipe | *how the graph represents it* | ONNX sidecar (+ name suffix) | `opset`, `precision` / `precision_mode`, `deform_core` |
| 3 | Engine build | *how/where the graph is compiled* | engine sidecar | `trt_version`, `sm_arch`, `min/opt/max_batch`, `network_typing`, `tf32`, `max_aux_streams`, `cuda_graph_compat`, `onnx_sha256` |
| 4 | Runtime invocation | *how it is called* | nowhere — not identity | actual batch, thresholds, JSON/drawing, graph replay |

Classifying a change: classes, shapes or output meaning change → axis 1; the
ONNX differs but the expected detections are the same → axis 2; the same ONNX,
different engine compatibility or performance → axis 3; no rebuild needed →
axis 4.

## Name vocabulary

ONNX (release assets and `dfine export` defaults), each with a `.json` sidecar:

- `dfine_<size>_op19.onnx` — the FP32 opset-19 base (`dfine_<size>_op<N>` for a
  non-default opset; the suffix-less `dfine_<size>.onnx` is the cache-dir spelling).
- `dfine_<size>_slim.onnx` — the v0.3 production surgical-FP16 recipe.
- `dfine_<size>_fp16_st.onnx` — the v0.2 legacy decoder-FP32 tier
  (`--precision fp16-legacy`).

The suffix names the *recipe* (axis 2), nothing else. It deliberately does not
encode the opset or the checkpoint: those live in the sidecar, and the cache
holds one artifact per public name — the fingerprint disambiguates engines.
The two recipe suffixes are defined once, in `python/dfine/cli.py`
(`_SLIM_SUFFIX` / `_LEGACY_SUFFIX`).

Engines:

- Cache: `dfine_<model>_<precision>-<fingerprint>-b1-<opt>-<max>-sm<arch>-trt<ver>.engine`,
  assembled only by `_cache_engine_name()` in `python/dfine/cli.py`. Every field
  after the fingerprint is a label — the resolver reads the batch profile from
  the engine sidecar and falls back to the name only for pre-sidecar engines.
- Dev tree / `dfine_build`: `<onnx-stem>_<precision>.engine` (underscore; the
  pre-v0.3.2 C++ default was a hyphen no other tool recognized).
- Engine sidecar: `<engine>.json` (appended) or `<engine-stem>.json` — probed in
  that order by the C++ runtime and the CLI alike.

## Identity, from strongest to weakest

1. `fingerprint` — `sha256(ONNX bytes + ONNX sidecar bytes)[:12]`, embedded in
   cache engine names; binds an engine to the exact artifact it was built from.
2. `onnx_sha256` in the engine sidecar — the engine's source graph. The resolver
   checks it when binding a cached engine: a contradicting sidecar beats a
   matching filename. Engines built by the C++ `dfine_build` fallback carry no
   `onnx_sha256` (the tool has no hash dependency) — the resolver treats them as
   provenance-unverified, exactly like pre-sidecar engines.
3. `checkpoint_sha256` in the ONNX sidecar — the training-side provenance;
   `dfine export` warns before replacing a previous cache export, noting which
   checkpoint it came from.

## Compatibility

Nothing published is renamed: all v0.3.1 asset names stay valid. Pre-v0.3.1
cache entries resolve with a warning; engines without a sidecar simply fall back
to filename facts. Engine sidecars written by the builders stamp
`schema_version: 1`; fields are only ever added, never renamed or removed.
