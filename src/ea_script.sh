#!/usr/bin/env bash
set -euo pipefail

# ---------- tunables (override via env) ----------
JOBS="${JOBS:-$(nproc)}"                           # parallel workers
SEEDS="${SEEDS:-30}"                               # seeds per (instance,pop)
GENS="${GENS:-20000}"                              # generations per run
LOGDIR="${LOGDIR:-results/ea_variant_c}"           # curve output root
POPS="${POPS:-20 50 100 200}"                      # population sizes
INSTANCES="${INSTANCES:-eil51 eil76 eil101 st70 kroA100 kroC100 kroD100 lin105 pcb442 pr2392 usa13509}"

# evaluation options
EVAL_SCRIPT="${EVAL_SCRIPT:-evaluation.py}"        # the schedule+Min/Mean+checkpoints evaluator
EVAL_OUT="${EVAL_OUT:-results/EA_mutation_results_two.txt}"
CHECKPOINTS="${CHECKPOINTS:-2000,5000,10000,20000}"
REDUCER="${REDUCER:-mean}"                         # mean|min

# allow skipping the run step and only summarizing
SKIP_RUNS="${SKIP_RUNS:-0}"

# ---------- env prep ----------
if [[ -d ".venv" ]]; then source .venv/bin/activate; fi
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD"
mkdir -p "$LOGDIR" "$(dirname "$EVAL_OUT")"

# ---------- choose single-run driver ----------
SINGLE=""
if [[ -x extra_scripts/run_evol_alg_c.py ]]; then
  SINGLE="sextra_scriptscripts/run_evol_alg_c.py"
elif [[ -x extra_scripts/run_evol_alg_c.py ]]; then
  SINGLE="extra_scripts/run_evol_alg_c.py"
elif [[ -f extra_scripts/run_evol_alg_c.py ]]; then
  chmod +x extra_scripts/run_evol_alg_c.py
  SINGLE="extra_scripts/run_evol_alg_c.py"
elif [[ -f extra_scripts/run_evol_alg_c.py ]]; then
  chmod +x extra_scripts/run_evol_alg_c.py
  SINGLE="extra_scripts/run_evol_alg_c.py"
else
  echo "ERROR: could not find a single-run script (extra_scripts/run_evol_alg_c.py or extra_scripts/run_evol_alg_c.py)." >&2
  exit 1
fi

# ---------- run grid (parallel) ----------
if [[ "$SKIP_RUNS" != "1" ]]; then
  echo "[`date`] Launching EA-C grid runs → $LOGDIR"
  TMP=$(mktemp)
  for inst in $INSTANCES; do
    for pop in $POPS; do
      for seed in $(seq 1 "$SEEDS"); do
        echo "python3 $SINGLE --instance $inst --pop $pop --seed $seed --gens $GENS --logdir $LOGDIR --pd" >> "$TMP"
      done
    done
  done

  if command -v parallel >/dev/null 2>&1; then
    parallel -j "$JOBS" < "$TMP"
  else
    xargs -P "$JOBS" -I{} bash -lc "{}" < "$TMP"
  fi
  rm -f "$TMP"
  echo "[`date`] All EA-C runs finished."
else
  echo "[`date`] SKIP_RUNS=1 → skipping run phase."
fi

# ---------- summarize (same format as your evaluation.py) ----------
echo "[`date`] Summarizing to $EVAL_OUT"
python3 "$EVAL_SCRIPT" \
  --variant_dir "$LOGDIR" \
  --out "$EVAL_OUT" \
  --seeds "$SEEDS" \
  --checkpoints "$CHECKPOINTS" \
  --reducer "$REDUCER"

echo "[`date`] Done. Table saved at: $EVAL_OUT"
