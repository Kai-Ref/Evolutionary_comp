# evaluation.py — EA Variant C (mutation-only) with schedule + Min/Mean + checkpoints
import os, argparse
import numpy as np

DATASETS = ["eil51","eil76","eil101", "st70", "kroA100", "kroC100", "kroD100", "lin105", "pcb442", \
            "pr2392", "usa13509"]
POPS = [20, 50, 100, 200]

def curve_path(base: str, inst: str, pop: int, seed: int) -> str:
    return os.path.join(base, inst, f"pop_{pop}", f"seed_{seed}", "best_cost_per_generation.npy")

def parse_checkpoints(cp_str: str):
    return [int(x) for x in cp_str.split(",")] if cp_str else [2000,5000,10000,20000]

# ---- infer number of nodes to print instance-specific schedule ----
def tsp_num_nodes(inst: str, datasets_dir="datasets") -> int:
    path = os.path.join(datasets_dir, f"{inst}.tsp")
    n = 0
    in_nodes = False
    try:
        with open(path, "r") as f:
            for line in f:
                s = line.strip()
                if not s: continue
                if s.upper().startswith("DIMENSION"):
                    try: return int(s.split(":")[1])
                    except Exception: pass
                if s.upper() == "NODE_COORD_SECTION": in_nodes = True; continue
                if s.upper() == "EOF": break
                if in_nodes:
                    parts = s.split()
                    if len(parts) >= 3: n += 1
    except FileNotFoundError:
        pass
    return n

# ---- same size-aware schedules as your EA ----
def size_aware_weights(n: int):
    # (TwoOpt, Jump, Exchange)
    if n <= 100:   return (0.65, 0.25, 0.10)
    if n <= 500:   return (0.55, 0.25, 0.20)
    if n <= 3000:  return (0.45, 0.25, 0.30)
    return (0.30, 0.20, 0.50)

def size_aware_strength_probability(n: int):
    # P(apply 1/2/3 mutations)
    if n <= 100:   return (0.70, 0.20, 0.10)
    if n <= 500:   return (0.60, 0.30, 0.10)
    if n <= 3000:  return (0.50, 0.30, 0.20)
    return (0.40, 0.35, 0.25)

def read_checkpoints_and_final(base: str, inst: str, pop: int, seeds: int, checkpoints, reducer: str):
    """
    Returns:
      agg_checkpoints: np.array of len(checkpoints) aggregated across seeds
      used: number of seeds used
      min_final: min over seeds of each run's final best-so-far value (arr[-1])
      mean_final: mean over seeds of arr[-1]
    Partial runs are allowed (for checkpoints we use last available gen in that run).
    """
    rows, finals = [], []
    used = 0
    for seed in range(1, seeds + 1):
        fp = curve_path(base, inst, pop, seed)
        if not os.path.exists(fp):
            continue
        try:
            arr = np.load(fp)
            if arr.size == 0: continue
            # checkpoint values (fallback to last if run shorter)
            vals = [float(arr[min(g, len(arr)-1)]) for g in checkpoints]
            rows.append(vals)
            finals.append(float(arr[-1]))
            used += 1
        except Exception:
            continue
    if not rows:
        return None, 0, None, None
    A = np.vstack(rows)
    agg = A.mean(axis=0) if reducer == "mean" else A.min(axis=0)
    return agg, used, float(np.min(finals)), float(np.mean(finals))

def main():
    ap = argparse.ArgumentParser(description="Summarize EA mutation-only results (+schedule + Min/Mean).")
    ap.add_argument("--variant_dir", type=str, default="results/ea_variant_c")
    ap.add_argument("--out", type=str, default="results/EA_mutation_results_two.txt")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--checkpoints", type=str, default="2000,5000,10000,20000")
    ap.add_argument("--reducer", choices=["mean","min"], default="mean")
    args = ap.parse_args()

    checkpoints = parse_checkpoints(args.checkpoints)

    # Build header with schedule + Min/Mean + checkpoints
    cp_headers = [f"Gen={g}" for g in checkpoints]
    header = (
        f"{'Instance':<10} | {'Pop':>3} | {'Seeds':>5} | "
        f"{'W(T/J/E)':>10} | {'P(1/2/3)':>10} | {'Min':>10} | {'Mean':>10} | "
        + " | ".join(f"{h:>10}" for h in cp_headers)
    )
    SEP = "-" * len(header)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    lines = []
    lines.append(SEP)
    lines.append("Variant: EA (Mutation-only)")
    lines.append(f"Seeds requested: {args.seeds} | Reducer (checkpoints): {args.reducer}")
    lines.append(SEP)
    lines.append(header)
    lines.append(SEP)

    for inst in DATASETS:
        n = tsp_num_nodes(inst)
        wt = size_aware_weights(n)
        pr = size_aware_strength_probability(n)
        w_str = f"{wt[0]:.2f}/{wt[1]:.2f}/{wt[2]:.2f}"
        p_str = f"{pr[0]:.2f}/{pr[1]:.2f}/{pr[2]:.2f}"

        for pop in POPS:
            agg, used, min_final, mean_final = read_checkpoints_and_final(
                args.variant_dir, inst, pop, args.seeds, checkpoints, args.reducer
            )
            if agg is None:
                vals = " | ".join([f"{'NA':>10}"] * len(checkpoints))
                lines.append(f"{inst:<10} | {pop:>3} | {used:>5} | {w_str:>10} | {p_str:>10} | "
                             f"{'NA':>10} | {'NA':>10} | {vals}")
            else:
                cp_vals = " | ".join([f"{v:>10.2f}" for v in agg])
                lines.append(f"{inst:<10} | {pop:>3} | {used:>5} | {w_str:>10} | {p_str:>10} | "
                             f"{min_final:>10.2f} | {mean_final:>10.2f} | {cp_vals}")
        lines.append(SEP)

    report = "\n".join(lines)
    print(report)
    with open(args.out, "w") as f:
        f.write(report + "\n")
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
