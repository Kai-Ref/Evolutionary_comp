set -euo pipefail

JOBS="${JOBS:-$(nproc)}"
SEEDS="${SEEDS:-30}"
GENS="${GENS:-20000}"
LOGDIR="${LOGDIR:-results/ea_variant_c}"
POPS="${POPS:-20 50 100 200}"
INSTANCES="${INSTANCES:-eil51 eil76 eil101 st70 kroA100 kroC100 kroD100 lin105 pcb442 pr2392 usa13509}"

EVAL_SCRIPT="${EVAL_SCRIPT:-evaluation.py}"
EVAL_OUT="${EVAL_OUT:-results/EA_mutation_results_two.txt}"
CHECKPOINTS="${CHECKPOINTS:-2000,5000,10000,20000}"
REDUCER="${REDUCER:-mean}"

STRICT_LEN="${STRICT_LEN:-0}"     # 1 = skip only if arrays are long enough
SKIP_RUNS="${SKIP_RUNS:-0}"

if [[ -d ".venv" ]]; then source .venv/bin/activate; fi
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD"
mkdir -p "$LOGDIR" "$(dirname "$EVAL_OUT")"

SINGLE="extra_scripts/run_evol_alg_c.py"
[[ -f "$SINGLE" ]] || { echo "ERROR: missing $SINGLE" >&2; exit 1; }
chmod +x "$SINGLE" || true

# Helper: is a run already done?
is_done() {
  local inst="$1" pop="$2" seed="$3"
  local best="$LOGDIR/$inst/pop_${pop}/seed_${seed}/best_cost_per_generation.npy"
  local mean="$LOGDIR/$inst/pop_${pop}/seed_${seed}/mean_cost_per_generation.npy"
  [[ -f "$best" && -f "$mean" ]] || return 1
  if [[ "$STRICT_LEN" == "1" ]]; then
    python3 - "$best" "$mean" "$GENS" <<'PY'
import sys, numpy as np
best,mean,gens = sys.argv[1], sys.argv[2], int(sys.argv[3])
try:
    b = np.load(best, mmap_mode="r")
    m = np.load(mean, mmap_mode="r")
    sys.exit(0 if (len(b) >= gens+1 and len(m) >= gens+1) else 1)
except Exception:
    sys.exit(1)
PY
    return $?
  fi
  return 0
}

if [[ "$SKIP_RUNS" != "1" ]]; then
  echo "[`date`] Launching EA-C grid runs → $LOGDIR"
  TMP=$(mktemp)
  queued=0 skipped=0

  for inst in $INSTANCES; do
    for pop in $POPS; do
      for seed in $(seq 1 "$SEEDS"); do
        if is_done "$inst" "$pop" "$seed"; then
          echo "[skip] $inst pop=$pop seed=$seed (already have curves)"
          ((skipped++)) || true
        else
          echo "python3 $SINGLE --instance $inst --pop $pop --seed $seed --gens $GENS --logdir $LOGDIR --pd --skip-existing $( [[ $STRICT_LEN == 1 ]] && echo --strict-len )" >> "$TMP"
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

echo "[`date`] Summarizing to $EVAL_OUT"
python3 "$EVAL_SCRIPT" \
  --variant_dir "$LOGDIR" \
  --out "$EVAL_OUT" \
  --seeds "$SEEDS" \
  --checkpoints "$CHECKPOINTS" \
  --reducer "$REDUCER"
echo "[`date`] Done. Table saved at: $EVAL_OUT"
