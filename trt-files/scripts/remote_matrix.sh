#!/usr/bin/env bash
# remote_matrix.sh — the validation matrix on a subset, sized for a rented GPU box.
#
# Exercises every export/conversion/build tier the project ships on one or two
# model sizes and packs the evidence (bench numbers, predict parity, subset mAP,
# the official validation report) into a single tarball to take home.
# Budget with defaults: ~1.5-2 h on an RTX 3090-class card.
#
# Fresh-box setup (Ubuntu 22.04+, NVIDIA driver preinstalled by the provider):
#   sudo apt-get update && sudo apt-get install -y \
#       build-essential cmake git curl unzip nlohmann-json3-dev
#   curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
#   bash Miniconda3-latest-Linux-x86_64.sh -b
#   ~/miniconda3/bin/conda create -y -n dfine python=3.11 'cuda-toolkit>=12.4' -c nvidia
#       # Blackwell (RTX 50xx, SM 12.x) needs 'cuda-toolkit>=12.8'
#   source ~/miniconda3/bin/activate dfine
#   git clone https://github.com/PogChamper/dfine-cpp && cd dfine-cpp
#   pip install -e ".[gpu]"
#   # Only for the export stage (SKIP_EXPORT=0):
#   pip install torch --index-url https://download.pytorch.org/whl/cu128
#   git clone https://github.com/ArgoHA/D-FINE-seg "$HOME/D-FINE-seg"
#
# Then:  bash trt-files/scripts/remote_matrix.sh
#
# Toggles (env):
#   SIZES="n m"       model sizes to run (n is fast, m matches the README numbers)
#   COCO_LIMIT=500    subset size for eval/calibration (0 skips the image download)
#   RELEASE_TAG=v0.3.2
#   SKIP_EXPORT=0     1 = skip checkpoint download + export + conversion repro
#   SKIP_INT8=0       1 = skip the INT8-QDQ conversion/build (research tier)
#   SKIP_EVAL=0       1 = skip subset coco_eval
#   DFINE_SEG_DIR=~/D-FINE-seg
#   WORK=~/dfine-matrix
set -euo pipefail

SIZES="${SIZES:-n m}"
COCO_LIMIT="${COCO_LIMIT:-500}"
RELEASE_TAG="${RELEASE_TAG:-v0.3.2}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"
SKIP_INT8="${SKIP_INT8:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
DFINE_SEG_DIR="${DFINE_SEG_DIR:-$HOME/D-FINE-seg}"
WORK="${WORK:-$HOME/dfine-matrix}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPTS="$REPO/trt-files/scripts"
ASSETS="$WORK/assets" ENG="$WORK/engines" EXP="$WORK/exports" OUT="$WORK/out"
COCO="$WORK/coco"
mkdir -p "$ASSETS" "$ENG" "$EXP" "$OUT" "$COCO"
SUMMARY="$OUT/summary.txt"
: > "$SUMMARY"

say() { printf '\n=== [%s] %s ===\n' "$(date +%H:%M:%S)" "$*" | tee -a "$SUMMARY"; }
note() { printf '%s\n' "$*" | tee -a "$SUMMARY"; }

# --------------------------------------------------------------------------- #
say "preflight"
command -v nvidia-smi >/dev/null || { echo "no nvidia-smi — GPU box required"; exit 1; }
CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
ARCH="${CAP/./}"
GPU="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
note "GPU: $GPU (SM $CAP)"
awk "BEGIN{exit !($CAP < 7.5)}" && { echo "SM $CAP < 7.5 — TensorRT 10 needs Turing or newer; rent a different box"; exit 1; }
command -v nvcc >/dev/null || { echo "nvcc not on PATH — activate the conda env (see header)"; exit 1; }
python - <<'EOF' || { echo "python deps missing — pip install -e '.[gpu]' (see header)"; exit 1; }
import tensorrt, cv2, pycocotools  # noqa: F401
EOF
TRT_LIBS="$(python -c 'import tensorrt_libs, os; print(os.path.dirname(tensorrt_libs.__file__))' 2>/dev/null || true)"
export LD_LIBRARY_PATH="${TRT_LIBS:+$TRT_LIBS:}${CONDA_PREFIX:+$CONDA_PREFIX/lib:}${LD_LIBRARY_PATH:-}"
note "TensorRT: $(python -c 'import tensorrt; print(tensorrt.__version__)')  nvcc: $(nvcc --version | grep -oP 'release \K[0-9.]+')"

# --------------------------------------------------------------------------- #
say "fetch release assets ($RELEASE_TAG)"
DL="https://github.com/PogChamper/dfine-cpp/releases/download/$RELEASE_TAG"
( cd "$ASSETS"
  curl -fsSLO "$DL/SHA256SUMS"
  for s in $SIZES; do
    for f in "dfine_${s}_op19.onnx" "dfine_${s}_op19.json" \
             "dfine_${s}_slim.onnx" "dfine_${s}_slim.json"; do
      [ -f "$f" ] || curl -fsSLO "$DL/$f"
      grep " $f\$" SHA256SUMS | sha256sum -c - >/dev/null
    done
  done )
note "assets verified against SHA256SUMS"
[ -f "$WORK/cats.jpg" ] || curl -fsSL -o "$WORK/cats.jpg" http://images.cocodataset.org/val2017/000000039769.jpg

if { [ "$SKIP_EVAL" != 1 ] || [ "$SKIP_INT8" != 1 ]; } && [ "$COCO_LIMIT" -gt 0 ]; then
  say "COCO val subset ($COCO_LIMIT images)"
  ANN="$COCO/instances_val2017.json"
  if [ ! -f "$ANN" ]; then
    curl -fsSL -o "$COCO/ann.zip" http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip -jo "$COCO/ann.zip" annotations/instances_val2017.json -d "$COCO" >/dev/null
    rm "$COCO/ann.zip"
  fi
  python - "$ANN" "$COCO/val2017" "$COCO_LIMIT" <<'EOF'
import json, sys, urllib.request
from pathlib import Path
ann, out, n = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
out.mkdir(exist_ok=True)
imgs = sorted(json.load(open(ann))["images"], key=lambda i: i["id"])[:n]
for i, im in enumerate(imgs):
    p = out / im["file_name"]
    if not p.exists():
        urllib.request.urlretrieve(f"http://images.cocodataset.org/val2017/{im['file_name']}", p)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{n}")
EOF
fi

# --------------------------------------------------------------------------- #
say "C++ build (CUDA_ARCH=$ARCH)"
( cd "$REPO" && CUDA_ARCH="$ARCH" ./build.sh ) >"$OUT/build_cpp.log" 2>&1
test -x "$REPO/build/dfine_bench" && test -x "$REPO/build/dfine_detect"
note "built: dfine_bench + dfine_detect OK"

# --------------------------------------------------------------------------- #
say "engines: fp32 / slim / slim-g0 per size"
build_engine() { # onnx out extra-args...
  local onnx="$1" out="$2"; shift 2
  python "$SCRIPTS/build_engine.py" --onnx "$onnx" --output "$out" \
    --no-tf32 --max-batch 8 "$@" >"$OUT/$(basename "$out" .engine).build.log" 2>&1
  note "  built $(basename "$out")"
}
for s in $SIZES; do
  build_engine "$ASSETS/dfine_${s}_op19.onnx" "$ENG/dfine_${s}_fp32.engine"
  build_engine "$ASSETS/dfine_${s}_slim.onnx" "$ENG/dfine_${s}_slim.engine" --strongly-typed
  build_engine "$ASSETS/dfine_${s}_slim.onnx" "$ENG/dfine_${s}_slim_g0.engine" \
    --strongly-typed --max-aux-streams 0 --cuda-graph
done

# --------------------------------------------------------------------------- #
say "ctest (CPU + GPU suites)"
FIRST_SIZE="${SIZES%% *}"
DFINE_TEST_ENGINE="$ENG/dfine_${FIRST_SIZE}_fp32.engine" \
  ctest --test-dir "$REPO/build" --output-on-failure 2>&1 | tee "$OUT/ctest.log" | tail -3 | tee -a "$SUMMARY"

# --------------------------------------------------------------------------- #
if [ "$SKIP_EXPORT" != 1 ]; then
  say "export + conversion reproducibility"
  if [ ! -d "$DFINE_SEG_DIR" ]; then
    note "  D-FINE-seg not found at $DFINE_SEG_DIR — skipping (set DFINE_SEG_DIR or SKIP_EXPORT=1)"
  else
    declare -A CKPT=( [n]=dfine_n_coco.pth [s]=dfine_s_obj2coco.pth [m]=dfine_m_obj2coco.pth \
                      [l]=dfine_l_obj2coco.pth [x]=dfine_x_obj2coco.pth )
    for s in $SIZES; do
      ck="$WORK/ckpt/${CKPT[$s]}"
      mkdir -p "$WORK/ckpt"
      [ -f "$ck" ] || curl -fsSL -o "$ck" \
        "https://github.com/Peterande/storage/releases/download/dfinev1.0/${CKPT[$s]}"
      python "$SCRIPTS/export_dfine_onnx.py" --model-name "$s" --checkpoint "$ck" \
        --dfine-src "$DFINE_SEG_DIR" --opset 19 \
        --output "$EXP/dfine_${s}_op19.onnx" >"$OUT/export_${s}.log" 2>&1
      python "$SCRIPTS/convert_fp16_surgical.py" --onnx "$EXP/dfine_${s}_op19.onnx" \
        --output "$EXP/dfine_${s}_slim.onnx" --slim >"$OUT/convert_slim_${s}.log" 2>&1
      python "$SCRIPTS/convert_fp16.py" --onnx "$EXP/dfine_${s}_op19.onnx" \
        --output "$EXP/dfine_${s}_fp16_st.onnx" >"$OUT/convert_legacy_${s}.log" 2>&1
      for f in "dfine_${s}_op19.onnx" "dfine_${s}_slim.onnx"; do
        if cmp -s "$EXP/$f" "$ASSETS/$f"; then verdict=BYTE-IDENTICAL; else verdict=DIFFERS; fi
        note "  $f vs release: $verdict"
      done
      build_engine "$EXP/dfine_${s}_fp16_st.onnx" "$ENG/dfine_${s}_fp16_st.engine" --strongly-typed
    done
  fi
fi

# --------------------------------------------------------------------------- #
if [ "$SKIP_INT8" != 1 ] && [ "$COCO_LIMIT" -gt 0 ]; then
  say "INT8-QDQ conversion + build (research tier, size $FIRST_SIZE)"
  if python "$SCRIPTS/convert_int8.py" --onnx "$ASSETS/dfine_${FIRST_SIZE}_op19.onnx" \
       --output "$EXP/dfine_${FIRST_SIZE}_int8_qdq.onnx" \
       --images "$COCO/val2017" --ann "$COCO/instances_val2017.json" \
       --num-calib "$(( COCO_LIMIT < 200 ? COCO_LIMIT : 200 ))" >"$OUT/convert_int8.log" 2>&1
  then
    build_engine "$EXP/dfine_${FIRST_SIZE}_int8_qdq.onnx" \
      "$ENG/dfine_${FIRST_SIZE}_int8.engine" --int8
  else
    note "  INT8 stage failed (research tier) — see $OUT/convert_int8.log"
  fi
fi

# --------------------------------------------------------------------------- #
say "bench (batches 1,8)"
printf '%-30s %10s %10s\n' engine "b1 img/s" "b8 img/s" | tee -a "$SUMMARY"
for e in "$ENG"/*.engine; do
  "$REPO/build/dfine_bench" --engine "$e" --batches 1,8 >"$OUT/$(basename "$e" .engine).bench.log" 2>&1 || { note "  $(basename "$e"): BENCH FAILED"; continue; }
  b1="$(awk '$1==1{print $8}' "$OUT/$(basename "$e" .engine).bench.log")"
  b8="$(awk '$1==8{print $8}' "$OUT/$(basename "$e" .engine).bench.log")"
  printf '%-30s %10s %10s\n' "$(basename "$e")" "$b1" "$b8" | tee -a "$SUMMARY"
done

# --------------------------------------------------------------------------- #
if printf '%s' "$SIZES" | grep -qw m; then
  say "predict parity (m slim, the README cats picture)"
  "$REPO/build/dfine_detect" --engine "$ENG/dfine_m_slim.engine" \
    --image "$WORK/cats.jpg" --threshold 0.5 >"$OUT/predict.log" 2>&1 || true
  cats="$(grep -c '\bcat\b' "$OUT/predict.log" || true)"
  remotes="$(grep -c '\bremote\b' "$OUT/predict.log" || true)"
  if [ "$cats" -ge 2 ] && [ "$remotes" -ge 2 ]; then
    note "  parity OK: ${cats}x cat, ${remotes}x remote (expected 2/2, see README)"
  else
    note "  parity SUSPECT: ${cats}x cat, ${remotes}x remote — inspect $OUT/predict.log"
  fi
fi

# --------------------------------------------------------------------------- #
if [ "$SKIP_EVAL" != 1 ] && [ "$COCO_LIMIT" -gt 0 ]; then
  say "subset mAP (engine backend, $COCO_LIMIT images — informational, not full-val)"
  for s in $SIZES; do
    for tier in fp32 slim; do
      python "$SCRIPTS/coco_eval.py" --backends engine --model-name "$s" \
        --engine "$ENG/dfine_${s}_${tier}.engine" --images "$COCO/val2017" \
        --ann "$COCO/instances_val2017.json" --limit "$COCO_LIMIT" \
        >"$OUT/eval_${s}_${tier}.log" 2>&1 || { note "  eval $s/$tier FAILED"; continue; }
      ap="$(grep -oP 'Average Precision.*IoU=0.50:0.95.*= \K[0-9.]+' "$OUT/eval_${s}_${tier}.log" | head -1)"
      note "  $s/$tier subset mAP: ${ap:-?}"
    done
  done
fi

# --------------------------------------------------------------------------- #
say "official validation report"
python "$SCRIPTS/validation_report.py" --onnx "$ASSETS/dfine_${FIRST_SIZE}_slim.onnx" \
  --check-sums "$ASSETS/SHA256SUMS" --out "$OUT/val-report" 2>&1 | tail -2

say "pack"
for e in "$ENG"/*.engine; do   # sidecars travel, engines stay (50 MB each, arch-specific)
  cp "${e%.engine}.json" "$OUT/" 2>/dev/null || cp "$e.json" "$OUT/" 2>/dev/null || true
done
TAR="$WORK/dfine-matrix-$(hostname)-$(date +%Y%m%d-%H%M).tgz"
tar czf "$TAR" -C "$WORK" out
say "DONE — take home: $TAR"
cat "$SUMMARY"
