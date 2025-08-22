from src.EvolutionaryAlgorithm import EvolutionaryAlgorithm
from src.operations.selection.Tournament import Tournament
from src.operations.selection.FitnessBased import FitnessBased
from src.operations.crossover.Order import Order
from src.operations.mutation.Exchange import Exchange

import random, numpy as np, os, sys, statistics as stats

def run_once(file_path, seed, selection_kind="tournament", pop=50, gens=200):
    rng = random.Random(seed)
    np.random.seed(seed)
    sel = Tournament(k=3, rng=rng) if selection_kind=="tournament" else FitnessBased(rng=rng)
    ea = EvolutionaryAlgorithm(
        filepath=file_path,
        population_size=pop,
        precompute_distances=True,
        selection=sel,
        crossover=Order(rng=rng),
        mutation=Exchange(),
        crossover_rate=0.9,
        mutation_rate=0.2,
        elitism_k=2,
        seed=seed,
        log_dir="results/ea_variant_a"
    )
    best = ea.solve(max_generations=gens)
    return best.fitness

if __name__ == "__main__":
    file_path = "datasets/eil51.tsp"
    seeds = [101, 202, 303]
    out = []
    for s in seeds:
        cost = run_once(file_path, s, selection_kind="tournament", pop=50, gens=200)
        print(f"seed {s}: best={cost:.3f}")
        out.append(cost)
    print(f"mean={stats.mean(out):.3f}, stdev={stats.pstdev(out):.3f}")
