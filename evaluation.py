# evaluation.py — summarize EA Variant C (mutation-only)
# Reads: results/ea_variant_c/<instance>/pop_<N>/seed_<seed>/best_cost_per_generation.npy
# Writes: results/EA_mutation_results.txt

import os, argparse
import numpy as np

DATASETS = ["eil51","eil76"]
POPS = [20, 50, 100, 200]
SEP = "-" * 112

def curve_path(base: str, inst: str, pop: int, seed: int) -> str:
    return os.path.join(base, inst, f"pop_{pop}", f"seed_{seed}", "best_cost_per_generation.npy")

def parse_checkpoints(cp_str: str):
    return [int(x) for x in cp_str.split(",")] if cp_str else [2000,5000,10000,20000]

def read_checkpoints(base: str, inst: str, pop: int, seeds: int, checkpoints, gens: int, reducer: str):
    """
    Aggregate values at checkpoints across seeds that exist.
    If a curve is shorter than 'gens', we take the last available value (so you can summarize partial runs).
    """
    rows, used = [], 0
    for seed in range(1, seeds + 1):
        fp = curve_path(base, inst, pop, seed)
        if not os.path.exists(fp):
            continue
        try:
            arr = np.load(fp)
            # pick value at each checkpoint, or last available if not yet reached
            vals = [float(arr[min(g, len(arr)-1)]) for g in checkpoints]
            rows.append(vals)
            used += 1
        except Exception:
            continue
    if not rows:
        return None, 0
    A = np.vstack(rows)
    agg = A.mean(axis=0) if reducer == "mean" else A.min(axis=0)
    return agg, used

def main():
    ap = argparse.ArgumentParser(description="Summarize EA mutation-only results.")
    ap.add_argument("--variant_dir", type=str, default="results/ea_variant_c",
                    help="Root directory where EA C saves runs.")
    ap.add_argument("--out", type=str, default="results/EA_mutation_results.txt",
                    help="Where to write the table.")
    ap.add_argument("--seeds", type=int, default=30,
                    help="How many seeds to aggregate (1..N). Missing seeds are ignored.")
    ap.add_argument("--gens", type=int, default=20000,
                    help="Target generations per run (for info only; partial runs still summarized).")
    ap.add_argument("--checkpoints", type=str, default="2000,5000,10000,20000",
                    help="Comma-separated checkpoints to report.")
    ap.add_argument("--reducer", choices=["mean","min"], default="mean",
                    help="Aggregate across seeds with mean (default) or min.")
    args = ap.parse_args()

    checkpoints = parse_checkpoints(args.checkpoints)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    header = f"{'Instance':<10} | {'Pop':>3} | {'Seeds':>5} | {'Gens':>5} | " + " | ".join([f'Gen={g:>6}' for g in checkpoints])

    lines = []
    lines.append(SEP)
    lines.append("Variant: EA (Mutation-only)")
    lines.append(f"Seeds requested: {args.seeds} | Reducer: {args.reducer} | Target gens per run: {args.gens}")
    lines.append(SEP)
    lines.append(header)
    lines.append(SEP)

    for inst in DATASETS:
        for pop in POPS:
            agg, used = read_checkpoints(args.variant_dir, inst, pop, args.seeds, checkpoints, args.gens, args.reducer)
            if agg is None:
                lines.append(f"{inst:<10} | {pop:>3} | {0:>5} | {args.gens:>5} | " + " | ".join([f'{"NA":>8}']*len(checkpoints)))
            else:
                vals = " | ".join([f"{v:>8.2f}" for v in agg])
                lines.append(f"{inst:<10} | {pop:>3} | {used:>5} | {args.gens:>5} | {vals}")
        lines.append(SEP)

    report = "\n".join(lines)
    print(report)
    with open(args.out, "w") as f:
        f.write(report + "\n")
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
