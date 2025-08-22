from __future__ import annotations

from src.TSP import TSP
from src.Population import Population
from src.Individual import Individual
from src.FileWriter import FileWriter

from src.operations.selection.Tournament import Tournament
from src.operations.selection.FitnessBased import FitnessBased  # optional
from src.operations.mutation.TwoOpt import TwoOpt
from src.operations.mutation.Jump import Jump
from src.operations.mutation.Exchange import Exchange

import numpy as np
import random
import os
from typing import Optional, Sequence, List, Tuple


class EvolutionaryAlgorithmC(TSP):
    """
    Variant C: Generational EA with elitism (mutation only)
    No crossover; Children are the clones of selected parents and mutated 1-3 times;
    Mutations chosen from TwoOpt, Jump and Exchange with weight sets derived from the local search findings 
    """

    def __init__(
        self,
        filepath: str,
        population_size: int = 100,
        distance_metric: str = "euclidean",
        precompute_distances: bool = True,
        # Operators
        selection: Optional[object] = None,      # Default: Tournament(k=3) 
        mutation_ops: Optional[Sequence[object]] = None,  # Default: TwoOpt(), Jump() and Exchange()
        # Hyperparameters
        mutation_rate: float = 1.0,              # Mutate every child since this is mutation-only EA
        elitism_k: int = 2,
        seed: Optional[int] = None,
        log_dir: str = "results/ea_variant_c",
        # mutation scheduling
        mutation_weights: Optional[Tuple[float, float, float]] = None,     # TwoOpt, Jump and Exchange
        mutation_strength_probs: Optional[Tuple[float, float, float]] = None,  # Probability of applying 1/2/3 mutations
    ):
        # Base problem
        super().__init__(
            filepath=filepath,
            distance_metric=distance_metric,
            precompute_distances=precompute_distances,
            population_size=population_size,
            mutation=None,
        )

        # RNG
        self.seed = seed
        self.rng = random.Random(seed)
        np.random.seed(seed if seed is not None else None)

        # Selection
        self.selection = selection or Tournament(k=3, rng=self.rng)

        # Mutation operators
        self.mutation_ops = list(mutation_ops) if mutation_ops is not None else [TwoOpt(), Jump(), Exchange()]
        self.op_names = [str(op) for op in self.mutation_ops]  # "TwoOpt", "Jump", "Exchange"

        # Size criteria defaults (from the local-search analysis)
        n = len(self.population.individuals[0].permutation)
        self.mutation_weights = mutation_weights or self.size_aware_weights(n)           # len=3
        self.mutation_strength_probs = mutation_strength_probs or self.size_aware_strength_probability(n)

        # Hyperparams
        self.mutation_rate = float(mutation_rate)
        self.elitism_k = int(elitism_k)

        # Logging
        self.file_writer = FileWriter()
        self.log_dir_base = log_dir

        # Calculating the initial population fitness
        for ind in self.population.individuals:
            if ind.fitness is None:
                ind.calculate_fitness()

    
    def size_aware_weights(self, n: int) -> Tuple[float, float, float]:
        """
        Order of weights is (TwoOpt, Jump, Exchange).

        From Local Search results:
        TwoOpt generally best on most datasets (small/medium);
        Jump beats Exchange on small datasets;
        Exchange surpasses Jump on the two largest datasets and performs best overall;
        """
        if n <= 100:               # eil51, eil76, eil101, st70, kro*100, lin105
            return (0.65, 0.25, 0.10)   # Strong 2-opt; jump second
        elif n <= 500:             # pcb442 
            return (0.55, 0.25, 0.20)
        elif n <= 3000:            # pr2392 etc.
            return (0.45, 0.25, 0.30)   # shift weight to exchange
        else:                       # usa13509 (huge): exchange > jump, 2-opt progress rare/slow
            return (0.30, 0.20, 0.50)

    def size_aware_strength_probability(self, n: int) -> Tuple[float, float, float]:
        """
        Probability of applying 1/2/3 mutations per child
        Use more multi-mutation on larger instances to help escape plateaus
        """
        if n <= 100:
            return (0.70, 0.20, 0.10)
        elif n <= 500:
            return (0.60, 0.30, 0.10)
        elif n <= 3000:
            return (0.50, 0.30, 0.20)
        else:
            return (0.40, 0.35, 0.25)

    def elitism(self, pop: Population, k: int) -> list[Individual]:
        return sorted(pop.individuals, key=lambda ind: ind.fitness)[:max(0, k)]

    def select_parents(self, pop: Population, n_pairs: int) -> list[tuple[Individual, Individual]]:
        pairs = []
        for _ in range(n_pairs):
            sel = self.selection(pop, 2)
            p1, p2 = sel.individuals[0], sel.individuals[1]
            pairs.append((p1, p2))
        return pairs

    def clone(self, ind: Individual) -> Individual:
        clone = Individual(number_of_nodes=None, tsp=ind.tsp, permutation=ind.permutation.copy())
        clone.fitness = ind.fitness
        return clone

    def apply_one_mutation(self, child: Individual) -> Individual:
        # choose operator by weights: (TwoOpt, Jump, Exchange)
        op = self.rng.choices(self.mutation_ops, weights=self.mutation_weights, k=1)[0]
        n = len(child.permutation)
        i, j = self.rng.sample(range(n), 2)
        mutated = op.mutate_individual(child, i, j, update_fitness=True)
        if mutated.fitness is None:
            mutated.calculate_fitness()
        return mutated

    def maybe_mutate(self, child: Individual) -> Individual:
        # mutate with probability mutation_rate, applying 1–3 ops
        if self.rng.random() >= self.mutation_rate:
            return child
        strength = self.rng.choices([1, 2, 3], weights=self.mutation_strength_probs, k=1)[0]
        for _ in range(strength):
            child = self.apply_one_mutation(child)
        return child

    def log_arrays(self, instance_name: str, seed: Optional[int], best_hist: list[float], mean_hist: list[float]):
        seed_folder = f"seed_{seed}" if seed is not None else "seed_none"
        folder = os.path.join(self.log_dir_base, instance_name, seed_folder)
        os.makedirs(folder, exist_ok=True)

        self.file_writer.file_path = folder
        self.file_writer(np.array(best_hist), "best_cost_per_generation.npy")
        self.file_writer(np.array(mean_hist), "mean_cost_per_generation.npy")

    ### Public API ###
    def solve(self, max_generations: int = 2000) -> Individual:
        """
        Runs Variant C (mutation-only) and returns the best individual found.
        """
        pop = self.population
        n = len(pop.individuals)
        instance_name = os.path.splitext(os.path.basename(self.filepath))[0]

        best_history: list[float] = []
        mean_history: list[float] = []

        # Initial stats
        best = min(pop.individuals, key=lambda ind: ind.fitness)
        best_history.append(best.fitness)
        mean_history.append(sum(ind.fitness for ind in pop.individuals) / n)

        # Generational loop
        for _ in range(int(max_generations)):
            elites = self.elitism(pop, self.elitism_k)

            # Children: select -> clone -> mutate (NO crossover)
            n_pairs = (n - len(elites)) // 2
            next_inds: list[Individual] = elites.copy()

            for (p1, p2) in self.select_parents(pop, n_pairs):
                c1 = self.clone(p1)
                c2 = self.clone(p2)
                c1 = self.maybe_mutate(c1)
                c2 = self.maybe_mutate(c2)
                next_inds.extend([c1, c2])

            # If odd, top up using selection
            while len(next_inds) < n:
                sel = self.selection(pop, 1)
                lone = sel.individuals[0]
                clone = self.clone(lone)
                clone = self.maybe_mutate(clone)
                next_inds.append(clone)

            # Replace generation
            pop.individuals = next_inds[:n]
            pop.population_size = n

            # Stats
            cur_best = min(pop.individuals, key=lambda ind: ind.fitness)
            best = cur_best if cur_best.fitness < best.fitness else best

            best_history.append(best.fitness)
            mean_history.append(sum(ind.fitness for ind in pop.individuals) / n)

        # Logs
        self.log_arrays(instance_name, self.seed, best_history, mean_history)
        return best
