from __future__ import annotations

import os
import random
from typing import Optional, Sequence, Tuple, List

import numpy as np

from src.EvolutionaryAlgorithm import EvolutionaryAlgorithm as EA
from src.Population import Population
from src.Individual import Individual

# Importing mutation operators
from src.operations.mutation.TwoOpt import TwoOpt
from src.operations.mutation.Jump import Jump
from src.operations.mutation.Exchange import Exchange


#  local (minimization) tournament selection
class _Selected:
    # Lightweight wrapper so caller functions can keep using .individuals 
    def __init__(self, individuals: List[Individual]):
        self.individuals = individuals


class MinTournament:
    # Minimization tournament selection for TSP (lower fitness is better in this case) 
    def __init__(self, k: int = 3, rng: Optional[random.Random] = None):
        self.k = k
        self.rng = rng or random.Random()

    def __call__(self, population: Population, num_to_select: int) -> _Selected:
        inds = population.individuals
        winners: List[Individual] = []
        for _ in range(num_to_select):
            competitors = self.rng.sample(inds, self.k)
            winners.append(min(competitors, key=lambda ind: ind.fitness))  # MIN
        return _Selected(winners)


class EvolutionaryAlgorithm(EA):

    # Variant C: Generational GA with elitism, mutation-only
    # optional memetic polish + immigrant injection

    def __init__(
        self,
        filepath: str,
        population_size: int = 100,
        distance_metric: str = "euclidean",
        precompute_distances: bool = True,
        # operators
        selection: Optional[object] = None,                 # default: MinTournament(k=3)
        mutation_ops: Optional[Sequence[object]] = None,    # default: [TwoOpt, Jump, Exchange]
        # hyperparameters
        mutation_rate: float = 1.0,
        elitism_k: int = 2,
        seed: Optional[int] = None,
        log_dir: str = "results/ea_variant_c",
        # mutation scheduling
        mutation_weights: Optional[Tuple[float, float, float]] = None,        # (TwoOpt, Jump, Exchange)
        mutation_strength_probs: Optional[Tuple[float, float, float]] = None, # P(apply 1/2/3 mutations)
        # diversity / memetic
        immigrant_rate: float = 0.10,
        immigrant_period: int = 100,
        elite_memetic_frac: float = 0.10,
        memetic_budget: int = 10,
    ):
        # base problem
        super().__init__(
            filepath=filepath,
            distance_metric=distance_metric,
            precompute_distances=precompute_distances,
            population_size=population_size,
            selection=selection or MinTournament(k=3, rng=random.Random(seed)),
            crossover=None,                 # mutation-only: no crossover operations
            mutation=None,                  # override _maybe_mutate
            crossover_rate=0.0,
            mutation_rate=mutation_rate,
            elitism_k=elitism_k,
            seed=seed,
            log_dir=log_dir,
        )

        # RNG Setup
        self.seed = seed
        self.rng = random.Random(seed)
        if seed is not None:
            np.random.seed(seed)

        # operators
        self.mutation_ops = list(mutation_ops) if mutation_ops is not None else [TwoOpt(), Jump(), Exchange()]
        n_nodes = len(self.population.individuals[0].permutation)
        self.mutation_weights = mutation_weights or self.size_aware_weights(n_nodes)
        self.mutation_strength_probs = mutation_strength_probs or self.size_aware_strength_probability(n_nodes)

        # hyperparameters
        self.mutation_rate = float(mutation_rate)
        self.elitism_k = int(elitism_k)

        # logging
        # (handled by base class)

        # Calculate initial fitness values
        for ind in self.population.individuals:
            if ind.fitness is None:
                ind.calculate_fitness()

        # diversity / memetic
        self.immigrant_rate = float(immigrant_rate)
        self.immigrant_period = int(immigrant_period)
        self.elite_memetic_frac = float(elite_memetic_frac)
        self.memetic_budget = int(memetic_budget)

    # size-aware schedules
    def size_aware_weights(self, n: int) -> Tuple[float, float, float]:
        if n <= 100:
            return (0.65, 0.25, 0.10)
        elif n <= 500:
            return (0.55, 0.25, 0.20)
        elif n <= 3000:
            return (0.45, 0.25, 0.30)
        else:
            return (0.30, 0.20, 0.50)

    def size_aware_strength_probability(self, n: int) -> Tuple[float, float, float]:
        if n <= 100:
            return (0.70, 0.20, 0.10)
        elif n <= 500:
            return (0.60, 0.30, 0.10)
        elif n <= 3000:
            return (0.50, 0.30, 0.20)
        else:
            return (0.40, 0.35, 0.25)

    # overrides
    def _maybe_crossover(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        # mutation-only => clone parents
        return self._clone(p1), self._clone(p2)

    def _apply_one_mutation(self, child: Individual) -> Individual:
        op = self.rng.choices(self.mutation_ops, weights=self.mutation_weights, k=1)[0]
        n = len(child.permutation)
        i, j = self.rng.sample(range(n), 2)
        if isinstance(op, TwoOpt) and i > j:
            i, j = j, i
        mutated = op.mutate_individual(child, i, j, update_fitness=True)
        if mutated.fitness is None:
            mutated.calculate_fitness()
        return mutated

    def _maybe_mutate(self, child: Individual) -> Individual:
        if self.rng.random() >= self.mutation_rate:
            return child
        strength = self.rng.choices([1, 2, 3], weights=self.mutation_strength_probs, k=1)[0]
        for _ in range(strength):
            child = self._apply_one_mutation(child)
        return child

    def _memetic_polish(self, ind: Individual, attempts: int) -> Individual:
        two_opt = TwoOpt()
        cur = ind
        for _ in range(attempts):
            n = len(cur.permutation)
            i, j = sorted(self.rng.sample(range(n), 2))
            cand = two_opt.mutate_individual(cur, i, j, update_fitness=True)
            if cand.fitness < cur.fitness:
                cur = cand
        return cur

    def _inject_immigrants(self, pop: Population, frac: float) -> None:
        m = max(1, int(frac * len(pop.individuals)))
        pop.individuals.sort(key=lambda ind: ind.fitness)  # ascending
        n_nodes = len(pop.individuals[0].permutation)
        for idx in range(len(pop.individuals) - m, len(pop.individuals)):
            pop.individuals[idx] = Individual(number_of_nodes=n_nodes, tsp=self)  # random perm; fitness in ctor

    # public API 

    def solve(self, max_generations: int = 20000) -> Individual:
        # Runs Variant C and returns the best individual found.
        
        pop = self.population  # Population object
        n = len(pop.individuals)
        instance_name = os.path.splitext(os.path.basename(self.filepath))[0]

        best_history: List[float] = []
        mean_history: List[float] = []

        # initial stats
        best = min(pop.individuals, key=lambda ind: ind.fitness)
        best_history.append(best.fitness)
        mean_history.append(sum(ind.fitness for ind in pop.individuals) / n)

        # generational loop
        for gen in range(int(max_generations)):
            elites = self._elitism(pop, self.elitism_k)

            # creating children
            n_pairs = (n - len(elites)) // 2
            next_inds: List[Individual] = elites.copy()

            for (p1, p2) in self._select_parents(pop, n_pairs):
                (c1, c2) = self._maybe_crossover(p1, p2)  # clones
                c1 = self._maybe_mutate(c1)
                c2 = self._maybe_mutate(c2)
                next_inds.extend([c1, c2])

            # if population size is odd, top up using selection
            while len(next_inds) < n:
                lone = self.selection(pop, 1).individuals[0]
                clone = self._maybe_mutate(self._clone(lone))
                next_inds.append(clone)

            # memetic polish on top slice
            if self.elite_memetic_frac > 0 and self.memetic_budget > 0:
                polish_k = max(1, int(self.elite_memetic_frac * n))
                next_inds.sort(key=lambda ind: ind.fitness)
                for i in range(polish_k):
                    next_inds[i] = self._memetic_polish(next_inds[i], attempts=self.memetic_budget)

            # diversity injection every P generations
            if self.immigrant_period > 0 and (gen + 1) % self.immigrant_period == 0 and self.immigrant_rate > 0:
                tmp = Population(population_size=n, number_of_nodes=len(next_inds[0].permutation), tsp=self)
                tmp.individuals = next_inds
                self._inject_immigrants(tmp, self.immigrant_rate)
                next_inds = tmp.individuals

            # swap populations
            pop.individuals = next_inds[:n]
            pop.population_size = n

            # Record stats
            cur_best = min(pop.individuals, key=lambda ind: ind.fitness)
            best = cur_best if cur_best.fitness < best.fitness else best

            best_history.append(best.fitness)
            mean_history.append(sum(ind.fitness for ind in pop.individuals) / n)

            # optional progress logging
            # if gen in [2000, 5000, 10000, 20000]:
            #     print(f"{gen} mean: {best.fitness}")

        self._log_arrays(instance_name, self.seed, best_history, mean_history)
        return best
