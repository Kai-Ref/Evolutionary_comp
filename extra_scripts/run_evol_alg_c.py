import argparse, os
from src.EvolutionaryAlgorithmC import EvolutionaryAlgorithmC

def main():
    ap = argparse.ArgumentParser(description="Run one EA-C job (instance, pop, seed).")
    ap.add_argument("--instance", required=True, help="TSPLIB instance name (e.g., eil51)")
    ap.add_argument("--pop", type=int, required=True, help="Population size")
    ap.add_argument("--seed", type=int, required=True, help="Random seed")
    ap.add_argument("--gens", type=int, default=20000, help="Generations")
    ap.add_argument("--logdir", type=str, default="results/ea_variant_c", help="Output root")
    ap.add_argument("--pd", action="store_true", help="Precompute distances")
    args = ap.parse_args()

    ds_path = os.path.join("datasets", f"{args.instance}.tsp")
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
