import argparse, os, sys
import numpy as np
from src.EvolutionaryAlgorithmC import EvolutionaryAlgorithmC

def curve_paths(logdir, inst, pop, seed):
    base = os.path.join(logdir, inst, f"pop_{pop}", f"seed_{seed}")
    return (
        os.path.join(base, "best_cost_per_generation.npy"),
        os.path.join(base, "mean_cost_per_generation.npy"),
    )

def done_enough(best_path, mean_path, gens, strict_len: bool) -> bool:
    # Must exist
    if not (os.path.exists(best_path) and os.path.exists(mean_path)):
        return False
    if not strict_len:
        return True
    # Also verify arrays are long enough
    try:
        b = np.load(best_path, mmap_mode="r")
        m = np.load(mean_path, mmap_mode="r")
        return (len(b) >= gens + 1) and (len(m) >= gens + 1)
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser(description="Run one EA-C job (instance, pop, seed).")
    ap.add_argument("--instance", required=True, help="TSPLIB instance name (e.g., eil51)")
    ap.add_argument("--pop", type=int, required=True, help="Population size")
    ap.add_argument("--seed", type=int, required=True, help="Random seed")
    ap.add_argument("--gens", type=int, default=20000, help="Generations")
    ap.add_argument("--logdir", type=str, default="results/ea_variant_c", help="Output root")
    ap.add_argument("--pd", action="store_true", help="Precompute distances")
    # NEW:
    ap.add_argument("--skip-existing", action="store_true",
                    help="If outputs already exist (and optionally are long enough), do nothing.")
    ap.add_argument("--strict-len", action="store_true",
                    help="Skip only if arrays length >= gens+1.")
    args = ap.parse_args()

    # Skip if already done
    best_path, mean_path = curve_paths(args.logdir, args.instance, args.pop, args.seed)
    if args.skip_existing or args.strict_len:  # honor either flag being set
        if done_enough(best_path, mean_path, args.gens, args.strict_len):
            print(f"[skip] {args.instance} pop={args.pop} seed={args.seed} (exists)", flush=True)
            sys.exit(0)

    # Run EA-C
    ds_path = os.path.join("datasets", f"{args.instance}.tsp")
    os.makedirs(os.path.dirname(best_path), exist_ok=True)

    ea = EvolutionaryAlgorithmC(
        filepath=ds_path,
        population_size=args.pop,
        precompute_distances=args.pd,
        seed=args.seed,
        log_dir=args.logdir,
    )
    ea.solve(max_generations=args.gens)

if __name__ == "__main__":
    main()