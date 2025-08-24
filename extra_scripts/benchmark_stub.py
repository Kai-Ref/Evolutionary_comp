import sys, os, argparse, random, numpy as np, statistics as stats
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.EvolutionaryAlgorithmA import EvolutionaryAlgorithm
from src.EvolutionaryAlgorithmB import EvolutionaryAlgorithm as EvolutionaryAlgorithmB
from src.EvolutionaryAlgorithmC import EvolutionaryAlgorithm as EvolutionaryAlgorithmC

from src.operations.selection.Tournament import Tournament
from src.operations.selection.FitnessBased import FitnessBased
from src.operations.crossover.Order import Order
from src.operations.mutation.Exchange import Exchange

def run_once(file_path, seed, selection_kind, pop, gens):
    rng = random.Random(seed)
    np.random.seed(seed)
    # allow default: let the variant choose its own selector
    if selection_kind == "default":
        selection = None
    else:
        selection = Tournament() if selection_kind == "tournament" else FitnessBased()
    ea = EvolutionaryAlgorithm(
        filepath=file_path,
        population_size=pop,
        precompute_distances=True,
        selection=selection,        # None -> Variant A defaults to Tournament(k=3)
        crossover=Order(),          # same as A's default
        mutation=Exchange(),        # same as A's default
        crossover_rate=0.9,
        mutation_rate=0.2,
        elitism_k=2,
        seed=seed,
        log_dir="results/ea_variant_a"
    )
    best = ea.solve(max_generations=gens)
    return best.fitness

def run_once_b(file_path, seed, selection_kind, pop, gens):
    rng = random.Random(seed)
    np.random.seed(seed)
    if selection_kind == "default":
        selection = None            # Variant B defaults to FitnessBased(self.population, 3)
    else:
        selection = FitnessBased() if selection_kind == "fitness" else Tournament()
    ea = EvolutionaryAlgorithmB(
        filepath=file_path,
        population_size=pop,
        precompute_distances=True,
        selection=selection,
        # Variant B uses internal PMX/Cycle crossover, no mutation
        crossover_rate=0.65,
        elitism_k=2,
        seed=seed,
        log_dir="results/ea_variant_b"
    )
    best = ea.solve(max_generations=gens)
    return best.fitness

def run_once_c(file_path, seed, selection_kind, pop, gens):
    rng = random.Random(seed)
    np.random.seed(seed)
    # Variant C default is MinTournament(k=3); allow override to FitnessBased if asked
    if selection_kind == "default":
        selection = None
    else:
        selection = FitnessBased() if selection_kind == "fitness" else None
    ea = EvolutionaryAlgorithmC(
        filepath=file_path,
        population_size=pop,
        precompute_distances=True,
        selection=selection,
        # mutation-only defaults; keep explicit for clarity
        mutation_rate=1.0,
        elitism_k=2,
        seed=seed,
        log_dir="results/ea_variant_c"
    )
    best = ea.solve(max_generations=gens)
    return best.fitness

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instances", nargs="+", default=["datasets/eil51.tsp", "datasets/eil76.tsp", "datasets/lin105.tsp"])
    p.add_argument("--selection", choices=["default", "tournament", "fitness"], default="default")
    p.add_argument("--pop", type=int, default=50)
    p.add_argument("--gens", type=int, default=20000)
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--seed_base", type=int, default=500)
    args = p.parse_args()

    os.makedirs("results", exist_ok=True)
    out_path = "results/your_EA.txt"

    # EA-A
    print("EA-A:")
    for inst in args.instances:
        costs = []
        for r in range(args.runs):
            seed = args.seed_base + r
            cost = run_once(inst, seed, args.selection, args.pop, args.gens)
            print(f"{os.path.basename(inst)} run {r+1}/{args.runs} seed {seed}: best={cost:.3f}")
            costs.append(cost)

        mean = stats.mean(costs)
        std = stats.pstdev(costs)
        line = (
            f"instance={os.path.basename(inst).split('.')[0]}, "
            f"selection={args.selection}, pop={args.pop}, gens={args.gens}, "
            f"runs={args.runs}, mean={mean:.3f}, std={std:.3f}\n"
        )
        with open(out_path, "a") as f:
            f.write(line)
        print("Wrote:", line.strip())

    # EA-B
    print("\nEA-B:")
    for inst in args.instances:
        costs = []
        for r in range(args.runs):
            seed = args.seed_base + r
            cost = run_once_b(inst, seed, args.selection, args.pop, args.gens)
            print(f"{os.path.basename(inst)} run {r+1}/{args.runs} seed {seed}: best={cost:.3f}")
            costs.append(cost)

        mean = stats.mean(costs)
        std = stats.pstdev(costs)
        line = (
            f"instance={os.path.basename(inst).split('.')[0]}, "
            f"selection={args.selection}, pop={args.pop}, gens={args.gens}, "
            f"runs={args.runs}, mean={mean:.3f}, std={std:.3f}\n"
        )
        with open(out_path, "a") as f:
            f.write(line)
        print("Wrote:", line.strip())

    # EA-C
    print("\nEA-C:")
    for inst in args.instances:
        costs = []
        for r in range(args.runs):
            seed = args.seed_base + r
            cost = run_once_c(inst, seed, args.selection, args.pop, args.gens)
            print(f"{os.path.basename(inst)} run {r+1}/{args.runs} seed {seed}: best={cost:.3f}")
            costs.append(cost)

        mean = stats.mean(costs)
        std = stats.pstdev(costs)
        line = (
            f"instance={os.path.basename(inst).split('.')[0]}, "
            f"selection={args.selection}, pop={args.pop}, gens={args.gens}, runs={args.runs}, "
            f"mean={mean:.3f}, std={std:.3f}\n"
        )
        with open(out_path, "a") as f:
            f.write(line)
        print("Wrote:", line.strip())

if __name__ == "__main__":
    main()
