# evaluation.py — Summarize EA variants A/B/C over checkpoints using cached .npy curves
import os, argparse
import numpy as np

# Instances + populations to sweep
DATASETS = ["eil51","eil76","eil101","st70","kroA100","kroC100","kroD100","lin105","pcb442","pr2392","usa13509"]
POPS     = [20, 50, 100, 200]

# --------------------- utilities ---------------------

def parse_checkpoints(cp_str: str):
    return [int(x) for x in cp_str.split(",")] if cp_str else [2000, 5000, 10000, 20000]

def tsp_num_nodes(inst: str, datasets_dir="datasets") -> int:
    """Infer node count to show size-aware schedules (used for Variant C)."""
    path = os.path.join(datasets_dir, f"{inst}.tsp")
    n = 0; in_nodes = False
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

# Variant C schedules
def size_aware_weights(n: int):
    # (TwoOpt, Jump, Exchange)
    if n <= 100:   return (0.65, 0.25, 0.10)
    if n <= 500:   return (0.55, 0.25, 0.20)
    if n <= 3000:  return (0.45, 0.25, 0.30)
    return (0.30, 0.20, 0.50)

def size_aware_strength_probability(n: int):
    # P(1/2/3 mutations)
    if n <= 100:   return (0.70, 0.20, 0.10)
    if n <= 500:   return (0.60, 0.30, 0.10)
    if n <= 3000:  return (0.50, 0.30, 0.20)
    return (0.40, 0.35, 0.25)

def _candidate_best_paths(results_root: str, variant_dir: str, instance_name: str, pop: int, seed: int):
    """
    Try both old and new layouts + both filenames:
      Old: results/<variant_dir>/<inst>/pop_<pop>/seed_<seed>/(<inst>_best... or best_cost_per_generation.npy)
      New: results/<variant_dir>/<inst>/seed_<seed>/(<inst>_best... or best_cost_per_generation.npy)
    """
    base = os.path.join(results_root, variant_dir, instance_name)
    return [
        os.path.join(base, f"pop_{pop}", f"seed_{seed}", f"{instance_name}_best_cost_per_generation.npy"),
        os.path.join(base, f"pop_{pop}", f"seed_{seed}", "best_cost_per_generation.npy"),
        os.path.join(base, f"seed_{seed}", f"{instance_name}_best_cost_per_generation.npy"),
        os.path.join(base, f"seed_{seed}", "best_cost_per_generation.npy"),
    ]

def _load_cached_best(results_root: str, variant_dir: str, instance_name: str, pop: int, seed: int, gens: int):
    """Return last value at generation `gens` if a matching curve exists."""
    for p in _candidate_best_paths(results_root, variant_dir, instance_name, pop, seed):
        if os.path.exists(p):
            try:
                arr = np.load(p)
                if len(arr) >= gens + 1:
                    return float(arr[gens])
            except Exception:
                pass
    return None

def _read_checkpoints_and_final(results_root: str, variant_dir: str, inst: str, pop: int,
                                seed_start: int, seeds: int, checkpoints, reducer: str):
    rows, finals = [], []
    used = 0
    for seed in range(seed_start, seed_start + seeds):
        found = False
        for p in _candidate_best_paths(results_root, variant_dir, inst, pop, seed):
            if os.path.exists(p):
                try:
                    arr = np.load(p)
                    if arr.size == 0: break
                    vals = [float(arr[min(g, len(arr)-1)]) for g in checkpoints]
                    rows.append(vals)
                    finals.append(float(arr[-1]))
                    used += 1
                    found = True
                    break
                except Exception:
                    break
        if not found:
            continue
    if not rows:
        return None, 0, None, None
    A = np.vstack(rows)
    agg = A.mean(axis=0) if reducer == "mean" else A.min(axis=0)
    return agg, used, float(np.min(finals)), float(np.mean(finals))

# --------------------- main ---------------------

def main():
    ap = argparse.ArgumentParser(description="Summarize EA results (A, B, C) over checkpoints using cached curves.")
    ap.add_argument("--variants", nargs="+", choices=["a","b","c","all"], default=["all"])
    ap.add_argument("--results_root", type=str, default="results")
    ap.add_argument("--dir_a", type=str, default="ea_variant_a")
    ap.add_argument("--dir_b", type=str, default="ea_variant_b")
    ap.add_argument("--dir_c", type=str, default="ea_variant_c")
    ap.add_argument("--out", type=str, default="results/EA_summary.txt")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--seed_start", type=int, default=1)           # set to 500 if your logs use seed_500.. etc
    ap.add_argument("--checkpoints", type=str, default="2000,5000,10000,20000")
    ap.add_argument("--reducer", choices=["mean","min"], default="mean")
    args = ap.parse_args()

    checkpoints = parse_checkpoints(args.checkpoints)

    # Resolve which variants to include
    if "all" in args.variants:
        wanted = [("A", args.dir_a), ("B", args.dir_b), ("C", args.dir_c)]
    else:
        mapping = {"a":("A", args.dir_a), "b":("B", args.dir_b), "c":("C", args.dir_c)}
        wanted  = [mapping[v] for v in args.variants]

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
    lines.append(f"Seeds requested: {args.seeds} | Seed start: {args.seed_start} | Reducer(checkpoints): {args.reducer}")
    lines.append(SEP)
    lines.append(header)
    lines.append(SEP)

    for label, vdir in wanted:
        lines.append(f"Variant: EA-{label}")
        lines.append(SEP)

        for inst in DATASETS:
            # Schedules meaningful only for C; dash out for A/B
            if label == "C":
                n = tsp_num_nodes(inst)
                wt = size_aware_weights(n)
                pr = size_aware_strength_probability(n)
                w_str = f"{wt[0]:.2f}/{wt[1]:.2f}/{wt[2]:.2f}"
                p_str = f"{pr[0]:.2f}/{pr[1]:.2f}/{pr[2]:.2f}"
            else:
                w_str = "-" * 10
                p_str = "-" * 10

            for pop in POPS:
                agg, used, min_final, mean_final = _read_checkpoints_and_final(
                    args.results_root, vdir, inst, pop, args.seed_start, args.seeds, checkpoints, args.reducer
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
