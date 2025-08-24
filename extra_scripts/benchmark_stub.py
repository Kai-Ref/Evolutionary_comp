import sys, os, argparse, random, numpy as np, statistics as stats
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.EvolutionaryAlgorithmC import EvolutionaryAlgorithm as EvolutionaryAlgorithmC
from src.operations.selection.FitnessBased import FitnessBased

# Commented out for Evaluating EA variant C 
# from src.EvolutionaryAlgorithmA import EvolutionaryAlgorithm as EvolutionaryAlgorithmA
# from src.operations.selection.Tournament import Tournament
# from src.operations.crossover.Order import Order
# from src.operations.mutation.Exchange import Exchange


def _candidate_best_paths(results_root: str, instance_name: str, pop: int, seed: int):
    # Check places where results are saved
    base = os.path.join(results_root, "ea_variant_c", instance_name)
    paths = [
        os.path.join(base, f"pop_{pop}", f"seed_{seed}", f"{instance_name}_best_cost_per_generation.npy"),
        os.path.join(base, f"pop_{pop}", f"seed_{seed}", "best_cost_per_generation.npy"),
        os.path.join(base, f"seed_{seed}", f"{instance_name}_best_cost_per_generation.npy"),
        os.path.join(base, f"seed_{seed}", "best_cost_per_generation.npy"),
    ]
    return paths


def _load_cached_best(results_root: str, instance_name: str, pop: int, seed: int, gens: int):
    # Look for existing cached results to avoid re-running expensive experiments
    for p in _candidate_best_paths(results_root, instance_name, pop, seed):
        if os.path.exists(p):
            arr = np.load(p)
            if len(arr) > gens:  # Ensures enough generations as required
                return float(arr[gens])
    return None


def run_once_c(file_path, seed, selection_kind, pop, gens):
    random.seed(seed)
    np.random.seed(seed)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_root = os.path.join(repo_root, "results")

    instance_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Skip expensive computation if we already have results
    cached = _load_cached_best(results_root, instance_name, pop, seed, gens)
    if cached:
        print(f"[cache] C {instance_name} seed {seed}: best={cached:.3f}")
        return cached

    # Set up selection method
    selection = FitnessBased() if selection_kind == "fitness" else None

    ea = EvolutionaryAlgorithmC(
        filepath=file_path,
        population_size=pop,
        precompute_distances=True,
        selection=selection,
        mutation_rate=1.0,
        elitism_k=2,
        seed=seed,
        log_dir="results/ea_variant_c",
    )
    
    best = ea.solve(max_generations=gens)
    return best.fitness


def main():
    parser = argparse.ArgumentParser()
    
    # TSP instances for benchmarking
    default_instances = [
        "datasets/eil51.tsp", "datasets/eil76.tsp", "datasets/eil101.tsp",
        "datasets/st70.tsp", "datasets/kroA100.tsp", "datasets/kroC100.tsp", 
        "datasets/kroD100.tsp", "datasets/lin105.tsp", "datasets/pcb442.tsp",
        "datasets/pr2392.tsp", "datasets/usa13509.tsp"
    ]
    
    parser.add_argument("--instances", nargs="+", default=default_instances)
    parser.add_argument("--selection", choices=["default", "fitness"], default="default")
    parser.add_argument("--pop", type=int, default=50)
    parser.add_argument("--gens", type=int, default=20000)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed_base", type=int, default=1)
    args = parser.parse_args()

    # Set up output directory
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(repo_root, "results"), exist_ok=True)
    out_path = os.path.join(repo_root, "results", "your_EA.txt")
    print(f"[benchmark] writing to {out_path}")

    print("\nEA-C (best):")
    for inst in args.instances:
        costs = []
        for r in range(args.runs):
            seed = args.seed_base + r
            cost = run_once_c(inst, seed, args.selection, args.pop, args.gens)
            print(f"{os.path.basename(inst)} run {r+1}/{args.runs} seed {seed}: best={cost:.3f}")
            costs.append(cost)

        # Calculate stats and writes results
        mean = stats.mean(costs)
        std = stats.pstdev(costs) if len(costs) > 1 else 0.0
        
        result_line = (
            f"instance={os.path.basename(inst).split('.')[0]}, "
            f"selection={'fitness' if args.selection=='fitness' else 'default'}, "
            f"pop={args.pop}, gens={args.gens}, runs={args.runs}, "
            f"mean={mean:.3f}, std={std:.3f}\n"
        )
        
        with open(out_path, "a") as f:
            # Add a header if file is empty
            if os.path.getsize(out_path) == 0:
                f.write("EA-C (best):\n")
            f.write(result_line)
        
        print("Wrote:", result_line.strip())


if __name__ == "__main__":
    main()


### Previous benchmark.py (EA-A) commented out to produce result for the best EA (VariantC)
# import sys, os, argparse, random, numpy as np, statistics as stats
# sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
#
# from src.EvolutionaryAlgorithmA import EvolutionaryAlgorithm
# from src.operations.selection.Tournament import Tournament
# from src.operations.selection.FitnessBased import FitnessBased
# from src.operations.crossover.Order import Order
# from src.operations.mutation.Exchange import Exchange
#
# def run_once(file_path, seed, selection_kind, pop, gens):
#     rng = random.Random(seed)
#     np.random.seed(seed)
#     selection = Tournament(k=3, rng=rng) if selection_kind == "tournament" else FitnessBased(rng=rng)
#     ea = EvolutionaryAlgorithm(
#         filepath=file_path,
#         population_size=pop,
#         precompute_distances=True,
#         selection=selection,
#         crossover=Order(),
#         mutation=Exchange(),
#         crossover_rate=0.9,
#         mutation_rate=0.2,
#         elitism_k=2,
#         seed=seed,
#         log_dir="results/ea_variant_a"
#     )
#     best = ea.solve(max_generations=gens)
#     return best.fitness
#
# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("--instances", nargs="+", default=["datasets/eil51.tsp"])
#     p.add_argument("--selection", choices=["tournament","fitness"], default="tournament")
#     p.add_argument("--pop", type=int, default=50)
#     p.add_argument("--gens", type=int, default=10)
#     p.add_argument("--runs", type=int, default=1)
#     p.add_argument("--seed_base", type=int, default=100)
#     args = p.parse_args()
#
#     os.makedirs("results", exist_ok=True)
#     out_path = "results/your_EA.txt"
#
#     for inst in args.instances:
#         costs = []
#         for r in range(args.runs):
#             seed = args.seed_base + r
#             cost = run_once(inst, seed, args.selection, args.pop, args.gens)
#             print(f"{os.path.basename(inst)} run {r+1}/{args.runs} seed {seed}: best={cost:.3f}")
#             costs.append(cost)
#
#         mean = stats.mean(costs)
#         std = stats.pstdev(costs)
#         line = (
#             f"instance={os.path.basename(inst).split('.')[0]}, "
#             f"selection={args.selection}, pop={args.pop}, gens={args.gens}, "
#             f"runs={args.runs}, mean={mean:.3f}, std={std:.3f}\n"
#         )
#         with open(out_path, "a") as f:
#             f.write(line)
#         print("Wrote:", line.strip())
#
# if __name__ == "__main__":
#     main()
