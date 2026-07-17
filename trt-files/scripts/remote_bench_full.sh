#!/usr/bin/env bash
# remote_bench_full.sh — publication-methodology bench on a rented box.
#
# Runs AFTER remote_matrix.sh (reuses its $WORK layout: engines/, exports/,
# ckpt/, coco/). Adds the missing tiers (fast = slim + export sliders; max on m)
# and re-benches EVERYTHING with the reference methodology from the night
# matrix: batches 1/2/4/8 x 500 iters, 3 interleaved rounds (round outer,
# engines inner — clock drift hits every engine equally), medians, VRAM.
# mAP stays the 500-image subset. Budget: ~30-40 min on an RTX 3090.
#
# Toggles (env): SIZES="n m"  ROUNDS=3  ITERS=500  WARMUP=30
#                SKIP_FAST=0 (1 = no new exports, bench existing engines only)
#                WORK=~/dfine-matrix
set -euo pipefail

SIZES="${SIZES:-n m}"
ROUNDS="${ROUNDS:-3}"
ITERS="${ITERS:-500}"
WARMUP="${WARMUP:-30}"
SKIP_FAST="${SKIP_FAST:-0}"
WORK="${WORK:-$HOME/dfine-matrix}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPTS="$REPO/trt-files/scripts"
ENG="$WORK/engines" EXP="$WORK/exports" OUT="$WORK/out" COCO="$WORK/coco"
BLOG="$OUT/bench_full"
mkdir -p "$BLOG"

say() { printf '\n=== [%s] %s ===\n' "$(date +%H:%M:%S)" "$*"; }

TRT_LIBS="$(python -c 'import tensorrt_libs, os; print(os.path.dirname(tensorrt_libs.__file__))' 2>/dev/null || true)"
export LD_LIBRARY_PATH="${TRT_LIBS:+$TRT_LIBS:}${CONDA_PREFIX:+$CONDA_PREFIX/lib:}${LD_LIBRARY_PATH:-}"
test -x "$REPO/build/dfine_bench" || { echo "no build/dfine_bench — run remote_matrix.sh first"; exit 1; }

build_engine() { # onnx out extra-args...
  local onnx="$1" out="$2"; shift 2
  [ -f "$out" ] && { echo "  cached $(basename "$out")"; return 0; }
  python "$SCRIPTS/build_engine.py" --onnx "$onnx" --output "$out" \
    --no-tf32 --max-batch 8 "$@" >"$OUT/$(basename "$out" .engine).build.log" 2>&1
  echo "  built $(basename "$out")"
}

# --------------------------------------------------------------------------- #
if [ "$SKIP_FAST" != 1 ]; then
  say "fast/max tiers (export sliders -> surgical slim -> engine)"
  declare -A CKPT=( [n]=dfine_n_coco.pth [s]=dfine_s_obj2coco.pth [m]=dfine_m_obj2coco.pth \
                    [l]=dfine_l_obj2coco.pth [x]=dfine_x_obj2coco.pth )
  fast_tier() { # size out-tag extra-export-args... ; builds dfine_<size>_<tag>.engine
    local s="$1" tag="$2"; shift 2
    local base="$EXP/dfine_${s}_${tag}_op19.onnx" slim="$EXP/dfine_${s}_${tag}.onnx"
    local eng="$ENG/dfine_${s}_${tag}.engine"
    [ -f "$eng" ] && { echo "  cached $(basename "$eng")"; return 0; }
    python "$SCRIPTS/export_dfine_onnx.py" --model-name "$s" \
      --checkpoint "$WORK/ckpt/${CKPT[$s]}" --opset 19 \
      --num-queries 200 --cascade 1:100 "$@" \
      --output "$base" >"$OUT/export_${s}_${tag}.log" 2>&1
    python "$SCRIPTS/convert_fp16_surgical.py" --onnx "$base" --output "$slim" --slim \
      >"$OUT/convert_${s}_${tag}.log" 2>&1
    if [ "$tag" = max ]; then
      build_engine "$slim" "$eng" --strongly-typed --opt-batch 8
    else
      build_engine "$slim" "$eng" --strongly-typed
    fi
  }
  for s in $SIZES; do fast_tier "$s" fast; done
  if printf '%s' "$SIZES" | grep -qw m; then fast_tier m max --eval-idx 2; fi
fi

# --------------------------------------------------------------------------- #
say "interleaved bench: $ROUNDS rounds x $ITERS iters, batches 1,2,4,8"
ENGINES=()
for s in $SIZES; do
  for tier in fp16_st fp32 slim fast; do
    [ -f "$ENG/dfine_${s}_${tier}.engine" ] && ENGINES+=("$ENG/dfine_${s}_${tier}.engine")
  done
done
[ -f "$ENG/dfine_m_max.engine" ] && ENGINES+=("$ENG/dfine_m_max.engine")
echo "  ${#ENGINES[@]} engines: $(basename -a "${ENGINES[@]}" | tr '\n' ' ')"

for r in $(seq 1 "$ROUNDS"); do
  echo "  round $r/$ROUNDS"
  for e in "${ENGINES[@]}"; do
    "$REPO/build/dfine_bench" --engine "$e" --batches 1,2,4,8 \
      --warmup "$WARMUP" --iters "$ITERS" \
      >"$BLOG/r${r}_$(basename "$e" .engine).log" 2>&1
  done
done

# --------------------------------------------------------------------------- #
say "aggregate (medians across rounds) + VRAM"
python - "$BLOG" "$ROUNDS" <<'EOF' | tee "$OUT/bench_full_table.txt"
import re, statistics, sys
from pathlib import Path
blog, rounds = Path(sys.argv[1]), int(sys.argv[2])
# rows: batch p50 p90 p99 pre infer decode img/s ; plus "peak GPU mem ...: N MiB"
row = re.compile(r"^(\d+)\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)")
mem = re.compile(r"peak GPU mem[^:]*:\s*(\d+)\s*MiB")
TIER = {"fp16_st": "prod", "slim": "slim", "fp32": "fp32", "fast": "fast", "max": "max"}
data = {}  # (size, tier) -> {batch: ([p50...],[ips...]), "vram": [..]}
for log in sorted(blog.glob("r*_dfine_*.log")):
    m = re.match(r"r\d+_dfine_([a-z])_(\w+)$", log.stem)
    if not m:
        continue
    key = (m.group(1), TIER.get(m.group(2), m.group(2)))
    d = data.setdefault(key, {"vram": []})
    for line in log.read_text().splitlines():
        r = row.match(line)
        if r:
            b = int(r.group(1))
            d.setdefault(b, ([], []))
            d[b][0].append(float(r.group(2)))
            d[b][1].append(float(r.group(3)))
        v = mem.search(line)
        if v:
            d["vram"].append(int(v.group(1)))
order = {"prod": 0, "slim": 1, "fast": 2, "max": 3, "fp32": 4}
print(f"Medians of {rounds} interleaved rounds, p50 ms / img/s; VRAM = peak engine+buffers.\n")
print(f"{'size':<5}{'config':<10}{'b1':<13}{'b2':<13}{'b4':<13}{'b8':<13}{'VRAM MiB':<9}")
for (size, tier) in sorted(data, key=lambda k: (k[0], order.get(k[1], 9))):
    d = data[(size, tier)]
    cells = []
    for b in (1, 2, 4, 8):
        p50s, ipss = d.get(b, ([], []))
        cells.append(f"{statistics.median(p50s):.2f}/{statistics.median(ipss):.0f}"
                     if p50s else "-")
    vram = max(d["vram"]) if d["vram"] else 0
    print(f"{size:<5}{tier:<10}{cells[0]:<13}{cells[1]:<13}{cells[2]:<13}{cells[3]:<13}{vram:<9}")
EOF

# --------------------------------------------------------------------------- #
if [ -f "$COCO/instances_val2017.json" ]; then
  say "subset mAP for the new tiers"
  for e in "$ENG"/dfine_*_fast.engine "$ENG"/dfine_m_max.engine; do
    [ -f "$e" ] || continue
    name="$(basename "$e" .engine)"
    s="${name#dfine_}"; s="${s%%_*}"
    if python "$SCRIPTS/coco_eval.py" --backends engine --model-name "$s" \
         --engine "$e" --images "$COCO/val2017" --ann "$COCO/instances_val2017.json" \
         --limit 500 >"$OUT/eval_${name}.log" 2>&1; then
      ap="$(grep -oP 'Average Precision.*IoU=0.50:0.95.*= \K[0-9.]+' "$OUT/eval_${name}.log" | head -1)"
      echo "  $name subset mAP: ${ap:-?}"
    else
      echo "  $name eval FAILED — see $OUT/eval_${name}.log"
    fi
  done | tee -a "$OUT/bench_full_table.txt"
fi

say "pack"
TAR="$WORK/dfine-benchfull-$(hostname)-$(date +%Y%m%d-%H%M).tgz"
tar czf "$TAR" -C "$WORK" out
say "DONE — take home: $TAR"
