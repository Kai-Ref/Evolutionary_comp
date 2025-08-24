# top: already there
import sys, os, argparse, random, numpy as np, statistics as stats
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ... your imports for A/B/C + ops ...

def run_once(file_path, seed, selection_kind, pop, gens):
    rng = random.Random(seed)
    np.random.seed(seed)
    # Let Variant A use its internal default when selection_kind == "default"
    if selection_kind == "default":
        selection = None
    elif selection_kind == "tournament":
        # Your repo Tournament is maximization-based; better to use A's default
        selection = None
    else:  # "fitness"
        selection = FitnessBased()
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
        log_dir="results/ea_variant_a"
    )
    best = ea.solve(max_generations=gens)
    return best.fitness

def run_once_b(file_path, seed, selection_kind, pop, gens):
    rng = random.Random(seed)
    np.random.seed(seed)
    # Variant B default = FitnessBased; use None to let the variant set it
    selection = None if selection_kind == "default" else FitnessBased()
    ea = EvolutionaryAlgorithmB(
        filepath=file_path,
        population_size=pop,
        precompute_distances=True,
        selection=selection,
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
    # Variant C default = MinTournament; use None to let the variant set it
    selection = None if selection_kind == "default" else FitnessBased()
    ea = EvolutionaryAlgorithmC(
        filepath=file_path,
        population_size=pop,
        precompute_distances=True,
        selection=selection,
        mutation_rate=1.0,
        elitism_k=2,
        seed=seed,
        log_dir="results/ea_variant_c"
    )
    best = ea.solve(max_generations=gens)
    return best.fitness

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instances", nargs="+", default=[
        "datasets/eil51.tsp", "datasets/eil76.tsp", "datasets/lin105.tsp"
    ])
    p.add_argument("--selection", choices=["default", "tournament", "fitness"], default="default")
    p.add_argument("--pop", type=int, default=50)
    p.add_argument("--gens", type=int, default=20000)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--seed_base", type=int, default=500)
    args = p.parse_args()

    # Anchor results path to repo root and flush after writes
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(repo_root, "results"), exist_ok=True)
    out_path = os.path.join(repo_root, "results", "your_EA.txt")
    print(f"[benchmark] writing to {out_path}")

    # EA-A
    print("EA-A:")
    for inst in args.instances:
        costs = []
        for r in range(args.runs):
            seed = args.seed_base + r
            cost = run_once(inst, seed, args.selection, args.pop, args.gens)
            print(f"{os.path.basename(inst)} run {r+1}/{args.runs} seed {seed}: best={cost:.3f}")
            costs.append(cost)
        mean = stats.mean(costs); std = stats.pstdev(costs)
        line = (f"instance={os.path.basename(inst).split('.')[0]}, "
                f"selection={args.selection}, pop={args.pop}, gens={args.gens}, "
                f"runs={args.runs}, mean={mean:.3f}, std={std:.3f}\n")
        with open(out_path, "a") as f:
            f.write(line); f.flush(); os.fsync(f.fileno())
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
        mean = stats.mean(costs); std = stats.pstdev(costs)
        line = (f"instance={os.path.basename(inst).split('.')[0]}, "
                f"selection={args.selection}, pop={args.pop}, gens={args.gens}, "
                f"runs={args.runs}, mean={mean:.3f}, std={std:.3f}\n")
        with open(out_path, "a") as f:
            f.write(line); f.flush(); os.fsync(f.fileno())
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
        mean = stats.mean(costs); std = stats.pstdev(costs)
        line = (f"instance={os.path.basename(inst).split('.')[0]}, "
                f"selection={args.selection if args.selection=='fitness' else 'default'}, "
                f"pop={args.pop}, gens={args.gens}, runs={args.runs}, "
                f"mean={mean:.3f}, std={std:.3f}\n")
        with open(out_path, "a") as f:
            f.write(line); f.flush(); os.fsync(f.fileno())
        print("Wrote:", line.strip())

if __name__ == "__main__":
    main()
