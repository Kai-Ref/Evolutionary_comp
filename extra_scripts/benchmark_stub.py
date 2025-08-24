import sys, os, argparse, random, numpy as np, statistics as stats
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.EvolutionaryAlgorithmC import EvolutionaryAlgorithm as EvolutionaryAlgorithmC
from src.operations.selection.FitnessBased import FitnessBased
# from src.EvolutionaryAlgorithmA import EvolutionaryAlgorithm as EvolutionaryAlgorithmA
# from src.EvolutionaryAlgorithmB import EvolutionaryAlgorithm as EvolutionaryAlgorithmB
# from src.operations.selection.Tournament import Tournament
# from src.operations.crossover.Order import Order
# from src.operations.mutation.Exchange import Exchange


# ---------- cache helpers (handles both old and new folder/file layouts) ----------
def _candidate_best_paths(results_root: str, instance_name: str, pop: int, seed: int):
    """
    Return the possible .npy locations in priority order.
    Old layout:  results/ea_variant_c/<inst>/pop_<pop>/seed_<seed>/(best_cost_per_generation.npy or <inst>_best...)
    New layout:  results/ea_variant_c/<inst>/seed_<seed>/(best_cost_per_generation.npy or <inst>_best...)
    """
    base = os.path.join(results_root, "ea_variant_c", instance_name)
    paths = [
        # old layout (with pop_<pop>)
        os.path.join(base, f"pop_{pop}", f"seed_{seed}", f"{instance_name}_best_cost_per_generation.npy"),
        os.path.join(base, f"pop_{pop}", f"seed_{seed}", "best_cost_per_generation.npy"),
        # new layout (no pop folder)
        os.path.join(base, f"seed_{seed}", f"{instance_name}_best_cost_per_generation.npy"),
        os.path.join(base, f"seed_{seed}", "best_cost_per_generation.npy"),
    ]
    return paths


def _load_cached_best(results_root: str, instance_name: str, pop: int, seed: int, gens: int):
    """
    Try all candidate paths. If any has >= gens+1 entries, return the last entry (float).
    """
    for p in _candidate_best_paths(results_root, instance_name, pop, seed):
        if os.path.exists(p):
            try:
                arr = np.load(p)
                if len(arr) >= gens + 1:
                    return float(arr[gens])
            except Exception:
                pass
    return None


# --------------------------------- Variant C ---------------------------------
def run_once_c(file_path, seed, selection_kind, pop, gens):
    rng = random.Random(seed)
    np.random.seed(seed)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_root = os.path.join(repo_root, "results")

    # --- Skip recomputation if we already have full curves for this (instance, seed) ---
    instance_name = os.path.splitext(os.path.basename(file_path))[0]
    cached = _load_cached_best(results_root, instance_name, pop, seed, gens)
    if cached is not None:
        print(f"[cache] C {instance_name} seed {seed}: best={cached:.3f}")
        return cached
    # -------------------------------------------------------------------------------

    # Variant C default = MinTournament (inside C). If you pass "fitness", use FitnessBased instead.
    selection = None if selection_kind == "default" else FitnessBased()

    ea = EvolutionaryAlgorithmC(
        filepath=file_path,
        population_size=pop,
        precompute_distances=True,
        selection=selection,   # keep None to use MinTournament default in Variant C
        mutation_rate=1.0,     # mutation-only
        elitism_k=2,
        seed=seed,
        log_dir="results/ea_variant_c",
    )
    best = ea.solve(max_generations=gens)
    return best.fitness


# --------------------------------------- A/B kept here for reference ---------------------------------------
# def run_once(...): pass
# def run_once_b(...): pass


def main():
    p = argparse.ArgumentParser()
    # Full TSPlib list; change if you only want a subset
    p.add_argument(
        "--instances",
        nargs="+",
        default=[
            "datasets/eil51.tsp",
            "datasets/eil76.tsp",
            "datasets/eil101.tsp",
            "datasets/st70.tsp",
            "datasets/kroA100.tsp",
            "datasets/kroC100.tsp",
            "datasets/kroD100.tsp",
            "datasets/lin105.tsp",
            "datasets/pcb442.tsp",
            "datasets/pr2392.tsp",
            "datasets/usa13509.tsp",
        ],
    )
    # Variant C should use its default (MinTournament). Optionally allow "fitness" to try FitnessBased.
    p.add_argument("--selection", choices=["default", "fitness"], default="default")
    p.add_argument("--pop", type=int, default=50)
    p.add_argument("--gens", type=int, default=20000)
    p.add_argument("--runs", type=int, default=30)     # 30 runs as required
    p.add_argument("--seed_base", type=int, default=1) # your precomputed seeds are seed_1..seed_30
    args = p.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(repo_root, "results"), exist_ok=True)
    out_path = os.path.join(repo_root, "results", "your_EA.txt")
    print(f"[benchmark] writing to {out_path}")

    if not os.path.exists(out_path):
        open(out_path, "w").close()

    # ------------- EA-C (best) -------------
    print("\nEA-C (best):")
    for inst in args.instances:
        costs = []
        for r in range(args.runs):
            seed = args.seed_base + r
            cost = run_once_c(inst, seed, args.selection, args.pop, args.gens)
            print(f"{os.path.basename(inst)} run {r+1}/{args.runs} seed {seed}: best={cost:.3f}")
            costs.append(cost)

        mean = stats.mean(costs)
        std = stats.pstdev(costs) if len(costs) > 1 else 0.0
        line = (
            f"instance={os.path.basename(inst).split('.')[0]}, "
            f"selection={'fitness' if args.selection=='fitness' else 'default'}, "
            f"pop={args.pop}, gens={args.gens}, runs={args.runs}, "
            f"mean={mean:.3f}, std={std:.3f}\n"
        )
        with open(out_path, "a") as f:
            # add a small header the first time we write
            if os.path.getsize(out_path) == 0:
                f.write("EA-C (best):\n")
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        print("Wrote:", line.strip())


if __name__ == "__main__":
    main()
