import sys, os, argparse, random, numpy as np, statistics as stats
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.EvolutionaryAlgorithmA import EvolutionaryAlgorithm
from src.operations.selection.Tournament import Tournament
from src.operations.selection.FitnessBased import FitnessBased
from src.operations.crossover.Order import Order
from src.operations.mutation.Exchange import Exchange

def run_once(file_path, seed, selection_kind, pop, gens):
    rng = random.Random(seed)
    np.random.seed(seed)
    selection = Tournament() if selection_kind == "tournament" else FitnessBased()
    ea = EvolutionaryAlgorithm(
        filepath=file_path,
        population_size=pop,
        precompute_distances=True,
        selection=selection,
        crossover=Order(),
        mutation=Exchange(),
        crossover_rate=0.9,
        mutation_rate=0.2,
        elitism_k=2,
        seed=seed,
        log_dir="results/ea_variant_b"
    )
    best = ea.solve(max_generations=gens)
    return best.fitness

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instances", nargs="+", default=["datasets/eil51.tsp", "datasets/eil76.tsp"])
    p.add_argument("--selection", choices=["fitness"], default="fitness")
    p.add_argument("--pop", type=int, default=50)
    p.add_argument("--gens", type=int, default=20000)
    p.add_argument("--runs", type=int, default=2)
    p.add_argument("--seed_base", type=int, default=500)
    args = p.parse_args()

    os.makedirs("results", exist_ok=True)
    out_path = "results/your_EA.txt"

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

if __name__ == "__main__":
    main()
