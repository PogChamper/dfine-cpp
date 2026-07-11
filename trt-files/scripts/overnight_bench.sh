#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# D-FINE full cross-backend + CUDA-graph benchmark, all sizes n→x.
# Unattended/overnight: idempotent (skips built engines), robust (per-step timeout,
# continues on failure), writes results + a consolidated REPORT.md.
#
#   bash trt-files/scripts/overnight_bench.sh [OUTDIR]
#
# Per size it builds FP32 + strongly-typed FP16 (default 2-aux) + FP16 0-aux (graph),
# runs profile.py (torch/onnx/trt/cpp on full COCO val: latency, FPS, GPU mem, mAP),
# and dfine_bench --graph-compare on the 0-aux engine (real CUDA-graph impact).
# Read $OUTDIR/REPORT.md in the morning; $OUTDIR/run.log has the full trace.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
S=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$S/../.." && pwd)
SEG=${DFINE_SEG_DIR:-$REPO/../D-FINE-seg}
PY=${PYTHON:-$SEG/.venv/bin/python}
COCO_IMAGES=${COCO_IMAGES:?set COCO_IMAGES to COCO val2017}
COCO_ANN=${COCO_ANN:?set COCO_ANN to instances_val2017.json}
[ -x "$PY" ] || { echo "Python interpreter not found: $PY" >&2; exit 1; }
[ -d "$COCO_IMAGES" ] || { echo "COCO image directory not found: $COCO_IMAGES" >&2; exit 1; }
[ -f "$COCO_ANN" ] || { echo "COCO annotation file not found: $COCO_ANN" >&2; exit 1; }
TRTLIB=${TRTLIB:-$("$PY" -c 'import os, tensorrt_libs; print(os.path.dirname(tensorrt_libs.__file__))')}
[ -d "$TRTLIB" ] || { echo "TensorRT library directory not found: $TRTLIB" >&2; exit 1; }
cd "$REPO" || exit 1
export LD_LIBRARY_PATH="$TRTLIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1

OUTDIR=${1:-$HOME/dfine_bench_$(date +%Y%m%d_%H%M)}
mkdir -p "$OUTDIR"
LOG=$OUTDIR/run.log
REPORT=$OUTDIR/REPORT.md
SIZES=${SIZES:-"n s m l x"}
SUBSET_ARGS=${SUBSET_ARGS:-"--full"}   # override e.g. SUBSET_ARGS="--subset 2000" for a faster dry run
read -r -a SUBSET_ARGV <<< "$SUBSET_ARGS"
BATCHES=(1 2 4 8)
CM_LIMIT=${CM_LIMIT:-2000}
FAILURES=0

declare -A CKPT_NAME=( [n]=dfine_n_coco.pt [s]=dfine_s_obj2coco.pt \
                       [m]=dfine_m_obj2coco.pt [l]=dfine_l_obj2coco.pt [x]=dfine_x_obj2coco.pt )

log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
# run a step with a timeout; log + continue on failure (never abort the whole night)
step() { local t=$1; shift; local d="$1"; shift; log "▶ $d"; if timeout "$t" "$@" >>"$LOG" 2>&1; then log "  ✓ $d"; return 0; else local rc=$?; FAILURES=$((FAILURES + 1)); log "  ✗ $d (rc=$rc — continuing)"; return "$rc"; fi; }

log "=== D-FINE overnight benchmark → $OUTDIR ==="
log "sizes=[$SIZES]  dataset=[$SUBSET_ARGS]  batches=[${BATCHES[*]}]"
if ! nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | tee -a "$LOG"; then
  log "GPU query failed"
  exit 1
fi

for sz in $SIZES; do
  log "──────────── SIZE $sz ────────────"
  rm -f "$OUTDIR/cmp_${sz}_fp32.json" "$OUTDIR/cmp_${sz}_fp16.json" \
    "$OUTDIR/graph_${sz}.txt" "$OUTDIR/pipeline_${sz}.txt" "$OUTDIR/parity_${sz}.txt" \
    "$OUTDIR/dets_${sz}_fullgraph.json" "$OUTDIR/dets_${sz}_split.json" \
    "$OUTDIR"/map_"${sz}"_*.txt
  ck=$SEG/pretrained/${CKPT_NAME[$sz]}
  [ -f "$ck" ] || ck=$SEG/${CKPT_NAME[$sz]}
  if [ ! -f "$ck" ]; then FAILURES=$((FAILURES + 1)); log "  ✗ checkpoint missing: $ck — skipping $sz"; continue; fi
  ONNX=trt-files/onnx/dfine_${sz}.onnx
  ONNX16=trt-files/onnx/dfine_${sz}_fp16_st.onnx
  E32=trt-files/engines/dfine_${sz}_fp32.engine
  E16=trt-files/engines/dfine_${sz}_fp16_st.engine        # default 2-aux (throughput)
  E16G=trt-files/engines/dfine_${sz}_fp16_g0.engine       # 0-aux (CUDA-graph capturable)

  [ -f "$ONNX" ]   || step 900  "export $sz ONNX" "$PY" "$S/export_dfine_onnx.py" --model-name "$sz" --checkpoint "$ck" --output "$ONNX"
  [ -f "$E32" ]    || step 1500 "build $sz FP32" "$PY" "$S/build_engine.py" --no-tf32 --max-batch 8 --onnx "$ONNX" --output "$E32"
  [ -f "$ONNX16" ] || step 600  "convert $sz FP16 ONNX" "$PY" "$S/convert_fp16.py" --onnx "$ONNX" --output "$ONNX16"
  [ -f "$E16" ]    || step 1500 "build $sz FP16 (2-aux)" "$PY" "$S/build_engine.py" --strongly-typed --no-tf32 --max-batch 8 --onnx "$ONNX16" --output "$E16"
  [ -f "$E16G" ]   || step 1500 "build $sz FP16 (0-aux)" "$PY" "$S/build_engine.py" --strongly-typed --no-tf32 --max-batch 8 --max-aux-streams 0 --onnx "$ONNX16" --output "$E16G"

  if [ ! -f "$E32" ] || [ ! -f "$E16" ]; then FAILURES=$((FAILURES + 1)); log "  ✗ $sz: engines missing, skipping profiles"; continue; fi

  # cross-backend: torch/onnx/trt-fp32/cpp-fp32 (torch/onnx engine-independent, need per-size checkpoint/onnx)
  FP32_REPORT=$OUTDIR/cmp_${sz}_fp32.json
  rm -f "$FP32_REPORT"
  step 3600 "$sz profile FP32 (torch/onnx/trt/cpp)" \
    "$PY" "$S/profile.py" --backends torch onnx trt cpp --engine "$E32" \
        "${SUBSET_ARGV[@]}" --batches "${BATCHES[@]}" \
        --warmup 30 --iters 150 --model-name "$sz" --checkpoint "$ck" --dfine-src "$SEG" \
        --onnx "$ONNX" --images "$COCO_IMAGES" --ann "$COCO_ANN" \
        --workdir "$OUTDIR" --out "$FP32_REPORT"
  # trt-fp16 / cpp-fp16
  FP16_REPORT=$OUTDIR/cmp_${sz}_fp16.json
  rm -f "$FP16_REPORT"
  step 2400 "$sz profile FP16 (trt/cpp)" \
    "$PY" "$S/profile.py" --backends trt cpp --engine "$E16" \
        "${SUBSET_ARGV[@]}" --batches "${BATCHES[@]}" \
        --warmup 30 --iters 150 --images "$COCO_IMAGES" --ann "$COCO_ANN" \
        --workdir "$OUTDIR" --out "$FP16_REPORT"
  # rigorous CUDA-graph impact on the 0-aux engine
  if [ -f "$E16G" ]; then
    log "▶ $sz graph-compare (0-aux FP16)"
    if timeout 600 "$REPO/build/dfine_bench" --engine "$E16G" --batches 1,4,8 \
        --warmup 30 --iters 200 --graph-compare > "$OUTDIR/graph_${sz}.txt" 2>>"$LOG"; then
      log "  ✓ $sz graph-compare"
    else
      FAILURES=$((FAILURES + 1))
      log "  ✗ $sz graph-compare"
    fi
  fi

  # Full-pipeline graph: per-stage CPU cost (pack/dispatch/wait/decode) of the
  # split gpu-decode path vs the single-launch graph, + in-run byte parity and the
  # live-threshold probe. NSYS=1 additionally records an Nsight Systems trace
  # (cuda,nvtx,osrt) to verify visually that the CPU idles while the graph runs
  # and that no hidden cudaMalloc/sync appears inside the captured region.
  if [ -f "$E16G" ]; then
    log "▶ $sz pipeline-compare (0-aux FP16)"
    NSYS_PREFIX=()
    if [ "${NSYS:-0}" = "1" ] && command -v nsys >/dev/null 2>&1; then
      NSYS_PREFIX=(nsys profile -t "cuda,nvtx,osrt" --force-overwrite true \
        -o "$OUTDIR/nsys_pipeline_${sz}")
      log "  (nsys trace → $OUTDIR/nsys_pipeline_${sz}.nsys-rep)"
    fi
    if timeout 900 "${NSYS_PREFIX[@]}" "$REPO/build/dfine_bench" --engine "$E16G" \
        --batches 1,4,8 --warmup 30 --iters 200 --pipeline-compare \
        > "$OUTDIR/pipeline_${sz}.txt" 2>>"$LOG"; then
      log "  ✓ $sz pipeline-compare"
    else
      FAILURES=$((FAILURES + 1))
      log "  ✗ $sz pipeline-compare"
    fi
  fi

  # mAP config matrix on the 0-aux engine — every runtime iteration isolated:
  # cpu-decode -> gpu-decode -> frozen(+own-mem) -> frozen b8 -> full-graph.
  # The full-graph run uses the fixed-resolution regime (640x480 val2017 subset,
  # 1061 real images) and is byte-diffed against the identically-filtered split
  # path: identical files == the graph replays exactly the split kernels.
  if [ -f "$E16G" ] && [ "${CONFIG_MATRIX:-1}" = "1" ]; then
    run_map() {
      local name=$1; shift
      log "▶ $sz mAP config [$name]"
      if timeout 3600 "$PY" "$S/cpp_coco_eval.py" --engine "$E16G" \
           --images "$COCO_IMAGES" --ann "$COCO_ANN" --limit "$CM_LIMIT" "$@" \
           > "$OUTDIR/map_${sz}_${name}.txt" 2>>"$LOG"; then
        log "  ✓ $sz mAP [$name]: $(grep -o 'AP@\[.50:.95\]=[0-9.]*' "$OUTDIR/map_${sz}_${name}.txt" | tail -1)"
        return 0
      else
        local rc=$?
        FAILURES=$((FAILURES + 1))
        log "  ✗ $sz mAP [$name] (rc=$rc; see run.log)"
        return "$rc"
      fi
    }
    run_map cpu
    run_map gpudecode --gpu-decode
    run_map frozen    --gpu-decode --freeze --own-device-memory
    run_map frozen_b8 --gpu-decode --freeze --own-device-memory --batch 8
    fullgraph_dets="$OUTDIR/dets_${sz}_fullgraph.json"
    split_dets="$OUTDIR/dets_${sz}_split.json"
    parity_file="$OUTDIR/parity_${sz}.txt"
    rm -f "$fullgraph_dets" "$split_dets" "$parity_file"
    fullgraph_ok=0
    split_ok=0
    if run_map fullgraph --full-graph --own-device-memory --filter-res 640x480 \
         --limit 0 --out "$fullgraph_dets"; then
      fullgraph_ok=1
    fi
    if run_map split_filtered --gpu-decode --freeze --own-device-memory --filter-res 640x480 \
         --limit 0 --out "$split_dets"; then
      split_ok=1
    fi
    if (( fullgraph_ok == 0 || split_ok == 0 )); then
      log "  ✗ $sz byte-parity not evaluated: one or both mAP runs failed"
      echo FAIL > "$parity_file"
    elif cmp -s "$fullgraph_dets" "$split_dets"; then
      log "  ✓ $sz byte-parity: full-graph == split gpu-decode (640x480 subset)"
      echo PASS > "$parity_file"
    else
      FAILURES=$((FAILURES + 1))
      log "  ✗ $sz byte-parity failed (dets_${sz}_fullgraph.json != dets_${sz}_split.json)"
      echo FAIL > "$parity_file"
    fi
  fi
done

log "=== consolidating → $REPORT ==="
if OUTDIR="$OUTDIR" SIZES="$SIZES" SUBSET_ARGS="$SUBSET_ARGS" CM_LIMIT="$CM_LIMIT" "$PY" - <<'PYEOF'
import json, os, re, glob, datetime
OUT=os.environ["OUTDIR"]; SIZES=os.environ["SIZES"].split()
SUBSET_ARGS=os.environ["SUBSET_ARGS"].strip(); CM_LIMIT=int(os.environ["CM_LIMIT"])
NAMES={"n":"nano","s":"small","m":"medium","l":"large","x":"xlarge"}
def load(p):
    try: return json.load(open(os.path.join(OUT,p)))
    except Exception: return None
def cell(x,n=2): return (f"{x:.{n}f}" if isinstance(x,(int,float)) else "—")
def be(rep,name,b,key):
    if not rep: return None
    e=rep["backends"].get(name,{}); L=e.get("latency",{}); r=L.get(str(b)) or L.get(b) or {}
    if key=="ap":
        m=e.get("map"); return m.get("AP") if m else None
    return r.get(key)
L=[]
L.append(f"# D-FINE cross-backend benchmark — {datetime.date.today()}")
dataset = "Full COCO val2017 (5000 images)" if SUBSET_ARGS == "--full" else f"Selection: `{SUBSET_ARGS}`"
L.append(f"\n{dataset}. Latency = e2e p50 ms (preprocess+infer+decode); "
         "FPS = img/s; GPU MiB = peak inference footprint. *torch/onnx GPU mem is an in-process delta and "
         "understates their true footprint; compare FP32↔FP16 within a stack.*\n")
BK=[("PyTorch (FP32)","fp32","torch"),("ONNXRuntime-GPU (FP32)","fp32","onnx"),
    ("TensorRT FP32 (py)","fp32","trt"),("TensorRT FP16 (py)","fp16","trt"),
    ("C++ FP32","fp32","cpp"),("C++ FP16","fp16","cpp")]
for sz in SIZES:
    f32=load(f"cmp_{sz}_fp32.json"); f16=load(f"cmp_{sz}_fp16.json")
    if not (f32 or f16): continue
    L.append(f"\n## {NAMES.get(sz,sz)} ({sz})\n")
    L.append("| backend | e2e b1 | infer b1 | FPS b1 | FPS b8 | GPU MiB | mAP |")
    L.append("|---|---|---|---|---|---|---|")
    for label,prec,name in BK:
        rep=f16 if prec=="fp16" else f32
        L.append(f"| {label} | {cell(be(rep,name,1,'p50'))} | {cell(be(rep,name,1,'infer_p50'))} | "
                 f"{cell(be(rep,name,1,'img_per_s'),1)} | {cell(be(rep,name,8,'img_per_s'),1)} | "
                 f"{cell(be(rep,name,1,'gpu_mem_mib'),0)} | {cell(be(rep,name,1,'ap'),4)} |")
L.append("\n## CUDA-graph impact (0-aux FP16 engine, `dfine_bench --graph-compare`)\n")
L.append("Rigorous same-run interleaved (immune to clock drift). D-FINE is dispatch-bound at small batch: "
         "the graph replaces the CPU's per-kernel enqueueV3 launches with one `cudaGraphLaunch`.\n")
L.append("| size | b1 no-graph→graph | b1 Δ | b8 no-graph→graph | b8 Δ | enqueueV3 vs graphLaunch (CPU b1) |")
L.append("|---|---|---|---|---|---|")
rowre=re.compile(r"batch (\d+).*?full wall\s*:\s*no-graph\s+([\d.]+)\s+vs graph\s+([\d.]+)\s+\(Δ\s+[\d.-]+ ms,\s*([+\-\d.]+)%",re.S)
cpre=re.compile(r"batch (\d+).*?CPU dispatch\s*:\s*enqueueV3\s+([\d.]+)\s+vs graphLaunch\s+([\d.]+)",re.S)
for sz in SIZES:
    p=os.path.join(OUT,f"graph_{sz}.txt")
    if not os.path.exists(p): continue
    t=open(p).read()
    fw={m[0]:(m[1],m[2],m[3]) for m in rowre.findall(t)}
    cp={m[0]:(m[1],m[2]) for m in cpre.findall(t)}
    b1=fw.get("1"); b8=fw.get("8"); c1=cp.get("1")
    def f(x): return f"{float(x[0]):.2f}→{float(x[1]):.2f} ms" if x else "—"
    def d(x): return f"{x[2]}%" if x else "—"
    cpu=f"{float(c1[0]):.2f} vs {float(c1[1]):.2f} ms" if c1 else "—"
    L.append(f"| {NAMES.get(sz,sz)} | {f(b1)} | {d(b1)} | {f(b8)} | {d(b8)} | {cpu} |")
L.append("\n**Recommendation:** batch-1 streaming → build `--max-aux-streams 0` + `use_cuda_graph`; "
         "batch throughput → default (2-aux) engine.\n")

# Full-pipeline graph: per-stage CPU cost + parity (dfine_bench --pipeline-compare)
L.append("\n## Full-pipeline graph (`dfine_bench --pipeline-compare`, 0-aux FP16)\n")
L.append("Per-stage HOST (CPU) cost, split gpu-decode path vs one `cudaGraphLaunch` per frame. "
         "`parity` counts byte-identical iterations; the threshold probe verifies the score "
         "threshold stays a live per-call knob inside the captured graph.\n")
L.append("| size | batch | CPU split | CPU graph | CPU freed | wall Δ | parity |")
L.append("|---|---|---|---|---|---|---|")
prow=re.compile(r"batch (\d+) \((\d+) iters.*?CPU total\s+([\d.]+)\s+([\d.]+)\s+([+\-\d.]+).*?"
                r"total wall\s+([\d.]+)\s+([\d.]+)\s+[+\-\d.]+ \(([+\-\d.]+)%\).*?"
                r"parity: (\d+)/(\d+) iterations mismatched",re.S)
for sz in SIZES:
    p=os.path.join(OUT,f"pipeline_{sz}.txt")
    if not os.path.exists(p): continue
    for m in prow.findall(open(p).read()):
        b,it,cs,cg,cf,ws,wg,wd,mm,tot=m
        ok = "OK" if mm=="0" else f"**{mm}/{tot} MISMATCH**"
        L.append(f"| {NAMES.get(sz,sz)} | {b} | {float(cs):.3f} ms | {float(cg):.3f} ms | "
                 f"{float(cf):+.3f} ms | {wd}% | {ok} |")

# mAP config matrix: every runtime iteration isolated
L.append("\n## mAP config matrix (0-aux FP16 engine, `cpp_coco_eval.py`)\n")
limit = "all sorted images" if CM_LIMIT == 0 else f"the first {CM_LIMIT} sorted images"
L.append(f"CPU/GPU/frozen columns use {limit}. Each runtime feature is isolated on the same engine. "
         "`fullgraph`/`split_filtered` run the "
         "fixed-resolution regime (640x480 val2017 subset); byte-parity below diffs their raw "
         "detection files.\n")
L.append("| size | cpu-decode | gpu-decode | frozen | frozen b8 | full-graph* | split*| byte-parity |")
L.append("|---|---|---|---|---|---|---|---|")
apre=re.compile(r"AP@\[\.50:\.95\]=([\d.]+)")
for sz in SIZES:
    def ap(name):
        p=os.path.join(OUT,f"map_{sz}_{name}.txt")
        if not os.path.exists(p): return "—"
        m=apre.findall(open(p).read())
        return m[-1] if m else "—"
    par=os.path.join(OUT,f"parity_{sz}.txt")
    parity=open(par).read().strip() if os.path.exists(par) else "—"
    row=[ap("cpu"),ap("gpudecode"),ap("frozen"),ap("frozen_b8"),ap("fullgraph"),ap("split_filtered")]
    if all(v=="—" for v in row): continue
    L.append(f"| {NAMES.get(sz,sz)} | "+" | ".join(row)+f" | {parity} |")
L.append("\n\\* 640x480-only subset — compare full-graph vs split within the column pair, not "
         "against the unfiltered columns.\n")
open(os.path.join(OUT,"REPORT.md"),"w").write("\n".join(L)+"\n")
print("wrote REPORT.md")
PYEOF
then
  :
else
  FAILURES=$((FAILURES + 1))
  log "Report consolidation failed"
fi
if (( FAILURES > 0 )); then
  log "=== DONE WITH $FAILURES FAILURE(S). Report: $REPORT ==="
  exit 1
fi
log "=== DONE. Report: $REPORT ==="
