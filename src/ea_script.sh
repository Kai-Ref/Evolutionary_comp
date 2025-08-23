set -euo pipefail

# ---------- tunables (override via env) ----------
JOBS="${JOBS:-$(nproc)}"                           # parallel workers
SEEDS="${SEEDS:-30}"                               # seeds per (instance,pop)
GENS="${GENS:-20000}"                              # generations per run
LOGDIR="${LOGDIR:-results/ea_variant_c}"           # curve output root
POPS="${POPS:-20 50 100 200}"                      # population sizes
INSTANCES="${INSTANCES:-eil51 eil76 eil101 st70 kroA100 kroC100 kroD100 lin105 pcb442 pr2392 usa13509}"

# evaluation options
EVAL_SCRIPT="${EVAL_SCRIPT:-evaluation.py}"        # your evaluator
EVAL_OUT="${EVAL_OUT:-results/EA_mutation_results_two.txt}"
CHECKPOINTS="${CHECKPOINTS:-2000,5000,10000,20000}"
REDUCER="${REDUCER:-mean}"                         # mean|min

# optional: set to 1 to *also* verify the npy length >= GENS+1 before skipping
STRICT_LEN="${STRICT_LEN:-0}"

# allow skipping the run step and only summarizing
SKIP_RUNS="${SKIP_RUNS:-0}"

# ---------- env prep ----------
if [[ -d ".venv" ]]; then source .venv/bin/activate; fi
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD"
mkdir -p "$LOGDIR" "$(dirname "$EVAL_OUT")"

# ---------- single-run driver ----------
SINGLE="extra_scripts/run_evol_alg_c.py"
if [[ ! -f "$SINGLE" ]]; then
  echo "ERROR: missing $SINGLE" >&2
  exit 1
fi
chmod +x "$SINGLE" || true

# helper: check if a run already “done”
is_done() {
  local inst="$1" pop="$2" seed="$3"
  local path="$LOGDIR/$inst/pop_${pop}/seed_${seed}/best_cost_per_generation.npy"
  if [[ ! -f "$path" ]]; then
    return 1   # not done
  fi
  if [[ "$STRICT_LEN" == "1" ]]; then
    # verify array length >= GENS+1 (index 0 = gen 0)
    python3 - "$path" "$GENS" <<'PY'
import sys, numpy as np
p = sys.argv[1]; gens = int(sys.argv[2])
try:
    a = np.load(p, mmap_mode="r")
    sys.exit(0 if len(a) >= gens+1 else 1)
except Exception:
    sys.exit(1)
PY
    return $?  # 0 means done
  fi
  return 0     # exists ⇒ treat as done
}

# ---------- run grid (parallel) ----------
if [[ "$SKIP_RUNS" != "1" ]]; then
  echo "[`date`] Launching EA-C grid runs → $LOGDIR"
  TMP=$(mktemp)
  queued=0 skipped=0

  for inst in $INSTANCES; do
    for pop in $POPS; do
      for seed in $(seq 1 "$SEEDS"); do
        if is_done "$inst" "$pop" "$seed"; then
          echo "[skip] $inst pop=$pop seed=$seed (results exist)"
          ((skipped++)) || true
        else
          echo "python3 $SINGLE --instance $inst --pop $pop --seed $seed --gens $GENS --logdir $LOGDIR --pd --skip-existing" >> "$TMP"
          ((queued++)) || true
        fi
      done
    done
  done

  echo "[info] queued: $queued | skipped: $skipped"

  if [[ "$queued" -gt 0 ]]; then
    if command -v parallel >/dev/null 2>&1; then
      parallel -j "$JOBS" < "$TMP"
    else
      xargs -P "$JOBS" -I{} bash -lc "{}" < "$TMP"
    fi
  else
    echo "[info] nothing to run."
  fi
  rm -f "$TMP"
  echo "[`date`] Run phase complete."
else
  echo "[`date`] SKIP_RUNS=1 → skipping run phase."
fi

# ---------- summarize ----------
echo "[`date`] Summarizing to $EVAL_OUT"
python3 "$EVAL_SCRIPT" \
  --variant_dir "$LOGDIR" \
  --out "$EVAL_OUT" \
  --seeds "$SEEDS" \
  --checkpoints "$CHECKPOINTS" \
  --reducer "$REDUCER"

echo "[`date`] Done. Table saved at: $EVAL_OUT"
