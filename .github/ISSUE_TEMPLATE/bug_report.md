---
name: Bug report
about: Report a build, runtime, or accuracy (mAP) regression
title: "[BUG] "
labels: bug
assignees: ""
---

**Describe the bug**
A clear description of what's broken.

**Repro steps**
1. Command run (e.g. `./build.sh`, `dfine_build ...`, `profile.py ...`)
2. Inputs used (engine/onnx/checkpoint, image or dataset subset)
3. Observed vs expected output

**Environment**
- GPU / driver:
- CUDA version:
- TensorRT version:
- OS / compiler:
- Commit hash:

**Logs / output**
```
paste relevant build or runtime output here
```

**mAP impact (if applicable)**
If this touches export/build/decode, include the `profile.py` before/after AP table.
