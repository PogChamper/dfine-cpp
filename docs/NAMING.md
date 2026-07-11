# Artifact naming and identity

Filenames are labels. Artifact bytes establish identity; sidecars declare provenance and behavior. The
runtime treats TensorRT engine IO and profile data as execution truth.

## Vocabulary

| Term | Meaning |
|---|---|
| Checkpoint | Trained PyTorch weights |
| ONNX artifact | One ONNX graph and its same-stem JSON sidecar |
| Engine | Target-local TensorRT plan; its sidecar is optional at runtime |
| Engine artifact | One engine and its matching engine sidecar |
| Runtime | `libdfine`, which executes an engine |
| Model pack | Published ONNX artifacts for one or more model sizes/recipes |
| Preset | Named graph/build recipe with an explicit accuracy/performance contract |

Do not use *model* alone when the distinction affects a command or compatibility decision.

## Identity axes

| Axis | Question | Source of truth |
|---|---|---|
| Model contract | What does the artifact compute? | Engine IO plus its matching ONNX or engine sidecar |
| Conversion recipe | How are graph operations and precision represented? | ONNX sidecar |
| Engine build | For which TensorRT/GPU/profile was it compiled? | Engine plus engine sidecar |
| Runtime invocation | How is this call executed? | Detector options and call arguments |

Examples:

- changing classes, input geometry, query count, or output meaning changes the model contract;
- changing FP32/FP16 placement while preserving detections changes the conversion recipe;
- changing TensorRT version, GPU architecture, auxiliary streams, or batch profile changes the engine build;
- changing threshold, actual batch, decode mode, or graph replay changes only runtime invocation.

## Published ONNX names

Every graph is published with a same-stem `.json` sidecar.

| Pattern | Recipe |
|---|---|
| `dfine_<size>_op19.onnx` | FP32 opset-19 base |
| `dfine_<size>_slim.onnx` | Production surgical-FP16 recipe |
| `dfine_<size>_fp16_st.onnx` | Legacy decoder-FP32 recipe |

`<size>` is `n`, `s`, `m`, `l`, or `x`. A suffix identifies a recipe; it does not identify the checkpoint, class set, or exact graph bytes. Those facts belong in the sidecar and hashes.

The active release model pack contains the `op19` and `slim` pairs. Legacy assets remain available from their original release but are not the default conversion path.

## Engine names

CLI cache entries use:

```text
dfine_<size>_<precision>-<fingerprint>-b1-<opt>-<max>[-g0]-sm<arch>-trt<version>.engine
```

`-g0` marks an engine built by `dfine build --cuda-graph`, with TensorRT auxiliary streams
disabled. The separate cache entry prevents one build policy from overwriting the other. The
runtime obtains the real batch profile from the engine and cross-checks engine-owned profile facts.

The maintained Python producers require `.onnx` graph inputs/outputs and `.engine` engine outputs;
those suffixes reserve unambiguous sidecar and staging namespaces. The runtime can open an explicitly
named engine with another suffix. A conventional local name is:

```text
<onnx-stem>.engine
```

Filenames do not establish identity, but move or rename each graph or engine together with its sidecar so discovery retains the complete contract.

## Fingerprints and hashes

The CLI cache fingerprint is:

```text
sha256(ONNX bytes + ONNX sidecar bytes)[:12]
```

It scopes a cache entry to the exact graph/sidecar pair. The production builder records the source graph as `onnx_sha256`; the resolver compares that value when both the engine candidate and source ONNX are available. The ONNX sidecar records the checkpoint hash and export provenance, including tool versions and model-source revision when available.

The trust chain is:

```text
checkpoint_sha256 + model_source + export recipe
                 ↓
       ONNX graph + sidecar
                 ↓ cache fingerprint / declared onnx_sha256
       TensorRT engine + sidecar
```

These hashes prevent accidental cache shadowing; they do not authenticate engine bytes or replace dataset validation.

## Sidecar locations

An ONNX artifact always uses:

```text
model.onnx
model.json
```

For an engine, the builder may use either:

```text
model.engine
model.engine.json    # appended form
```

or:

```text
model.engine
model.json           # same-stem form
```

The appended form is required when an ONNX graph and engine share a stem in the same directory; otherwise `model.json` belongs to the ONNX artifact. Runtime auto-discovery probes the appended form first, then the same-stem form. Passing an explicit metadata path disables discovery: that file must exist, parse, and agree with the engine.

Avoid keeping two engine sidecars next to one engine. Builders remove a stale shadowing twin when
publishing to an automatic-discovery location, while preserving the input ONNX sidecar.

## Sidecar responsibilities

The ONNX sidecar owns model and conversion facts:

- model size, task, input geometry, classes, labels, and query count;
- input/output names, shapes, box format, and score activation;
- preprocessing and resize geometry;
- checkpoint load status and SHA-256;
- model-source provenance, exporter/converter hashes, source-graph hash, simplification result, and
  tool versions;
- opset, deform core, precision, and precision recipe.

It carries `artifact_kind: "onnx"`. Its `max_batch` is an export-time build recommendation, not an
engine profile fact.

The production Python builder carries those fields forward and adds build facts:

- TensorRT version and GPU architecture;
- network typing and TF32 setting;
- min/opt/max batch;
- auxiliary-stream and graph-compatibility facts;
- source ONNX SHA-256.

Builders change `artifact_kind` to `"engine"`. The runtime cross-checks profile fields only for
engine metadata; the TensorRT profile remains authoritative. For legacy untagged metadata,
`trt_version` plus `min_batch` or `opt_batch` identifies engine-owned profile fields; `max_batch`
alone does not.

The FP32-only C++ builder records tensor/profile facts, TensorRT version, `tf32: false`, and
`max_aux_streams` (`0` for `--cuda-graph`, otherwise `null`). It uses weak network typing and omits
GPU architecture and source ONNX SHA-256. Treat such an engine as provenance-unverified unless it
is selected explicitly and validated separately.

The engine remains authoritative for tensor format, dimensions, data types, and optimization profile. A sidecar contradiction is an error, not an override.

## Cache resolution

Resolution follows artifact identity:

1. An explicit `--engine` wins; cache provenance resolution is bypassed.
2. Otherwise the resolved ONNX artifact determines the fingerprint.
3. The fingerprint selects cache candidates for that graph/sidecar pair.
4. A candidate whose sidecar records another ONNX hash is rejected.
5. An exact requested batch profile wins; otherwise the candidate with the largest maximum batch
   is chosen deterministically.

An absent source hash cannot be verified independently; an engine whose source ONNX is missing cannot be re-fingerprinted. The CLI reports the latter loss of provenance instead of presenting the result as verified.

## Schema evolution

ONNX and engine sidecars use `schema_version: 1` and identify their owner with `artifact_kind`.
Readers accept legacy sidecars without that field, accept absent optional fields, and reject a schema
newer than they support. Existing fields retain their meaning; compatible additions do not require
renaming artifacts.

Changes that alter tensor semantics, preprocessing, or output meaning require a new artifact contract and validation, even when the JSON schema can represent them without a version bump.
