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
set -u
REPO=/home/dxdxxd/projects/custom-dfine/D-FINE-cpp
SEG=/home/dxdxxd/projects/custom-dfine/D-FINE-seg
PY=$SEG/.venv/bin/python
S=$REPO/trt-files/scripts
cd "$REPO" || exit 1
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$SEG/.venv/lib/python3.11/site-packages/tensorrt_libs:/home/dxdxxd/miniconda3/lib
export PYTHONUNBUFFERED=1

OUTDIR=${1:-$HOME/dfine_bench_$(date +%Y%m%d_%H%M)}
mkdir -p "$OUTDIR"
LOG=$OUTDIR/run.log
REPORT=$OUTDIR/REPORT.md
SIZES=${SIZES:-"n s m l x"}
SUBSET_ARGS=${SUBSET_ARGS:-"--full"}   # override e.g. SUBSET_ARGS="--subset 2000" for a faster dry run
BATCHES="1 2 4 8"

declare -A CKPT=( [n]=$SEG/dfine_n_coco.pt [s]=$SEG/dfine_s_obj2coco.pt \
                  [m]=$SEG/pretrained/dfine_m_obj2coco.pt [l]=$SEG/dfine_l_obj2coco.pt [x]=$SEG/dfine_x_obj2coco.pt )

log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
# run a step with a timeout; log + continue on failure (never abort the whole night)
step() { local t=$1; shift; local d="$1"; shift; log "▶ $d"; if timeout "$t" "$@" >>"$LOG" 2>&1; then log "  ✓ $d"; else log "  ✗ $d (rc=$? — continuing)"; fi; }

log "=== D-FINE overnight benchmark → $OUTDIR ==="
log "sizes=[$SIZES]  dataset=[$SUBSET_ARGS]  batches=[$BATCHES]"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | tee -a "$LOG"

for sz in $SIZES; do
  log "──────────── SIZE $sz ────────────"
  ck=${CKPT[$sz]}
  if [ ! -f "$ck" ]; then log "  ✗ checkpoint missing: $ck — skipping $sz"; continue; fi
  ONNX=trt-files/onnx/dfine_${sz}.onnx
  ONNX16=trt-files/onnx/dfine_${sz}_fp16_st.onnx
  E32=trt-files/engines/dfine_${sz}_fp32.engine
  E16=trt-files/engines/dfine_${sz}_fp16_st.engine        # default 2-aux (throughput)
  E16G=trt-files/engines/dfine_${sz}_fp16_g0.engine       # 0-aux (CUDA-graph capturable)

  [ -f "$ONNX" ]   || step 900  "export $sz ONNX"        $PY $S/export_dfine_onnx.py --model-name $sz --checkpoint "$ck" --output $ONNX
  [ -f "$E32" ]    || step 1500 "build $sz FP32"          $PY $S/build_engine.py --no-tf32 --max-batch 8 --cuda-graph --onnx $ONNX --output $E32
  [ -f "$ONNX16" ] || step 600  "convert $sz FP16 ONNX"   $PY $S/convert_fp16.py --onnx $ONNX --output $ONNX16
  [ -f "$E16" ]    || step 1500 "build $sz FP16 (2-aux)"   $PY $S/build_engine.py --strongly-typed --no-tf32 --max-batch 8 --cuda-graph --onnx $ONNX16 --output $E16
  [ -f "$E16G" ]   || step 1500 "build $sz FP16 (0-aux)"   $PY $S/build_engine.py --strongly-typed --no-tf32 --max-batch 8 --cuda-graph --max-aux-streams 0 --onnx $ONNX16 --output $E16G

  if [ ! -f "$E32" ] || [ ! -f "$E16" ]; then log "  ✗ $sz: engines missing, skipping profiles"; continue; fi

  # cross-backend: torch/onnx/trt-fp32/cpp-fp32 (torch/onnx engine-independent, need per-size checkpoint/onnx)
  step 3600 "$sz profile FP32 (torch/onnx/trt/cpp)" \
    $PY $S/profile.py --backends torch onnx trt cpp --engine $E32 $SUBSET_ARGS --batches $BATCHES \
        --warmup 30 --iters 150 --model-name $sz --checkpoint "$ck" --onnx $ONNX \
        --workdir "$OUTDIR" --out "$OUTDIR/cmp_${sz}_fp32.json"
  # trt-fp16 / cpp-fp16
  step 2400 "$sz profile FP16 (trt/cpp)" \
    $PY $S/profile.py --backends trt cpp --engine $E16 $SUBSET_ARGS --batches $BATCHES \
        --warmup 30 --iters 150 --workdir "$OUTDIR" --out "$OUTDIR/cmp_${sz}_fp16.json"
  # rigorous CUDA-graph impact on the 0-aux engine
  if [ -f "$E16G" ]; then
    log "▶ $sz graph-compare (0-aux FP16)"
    timeout 600 "$REPO/build/dfine_bench" --engine $E16G --batches 1,4,8 --warmup 30 --iters 200 \
        --graph-compare > "$OUTDIR/graph_${sz}.txt" 2>>"$LOG" && log "  ✓ $sz graph-compare" || log "  ✗ $sz graph-compare"
  fi
done

log "=== consolidating → $REPORT ==="
OUTDIR="$OUTDIR" SIZES="$SIZES" $PY - <<'PYEOF'
import json, os, re, glob, datetime
OUT=os.environ["OUTDIR"]; SIZES=os.environ["SIZES"].split()
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
L.append("\nFull COCO val2017 (5000 imgs) unless noted. Latency = e2e p50 ms (preprocess+infer+decode); "
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
         "batch throughput → default (2-aux) engine. See docs/HANDOFF.md M2.2.\n")
open(os.path.join(OUT,"REPORT.md"),"w").write("\n".join(L)+"\n")
print("wrote REPORT.md")
PYEOF
log "=== DONE. Report: $REPORT ==="
