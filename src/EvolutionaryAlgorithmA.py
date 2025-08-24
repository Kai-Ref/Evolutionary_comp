from __future__ import annotations

import random
from typing import Optional

import numpy as np

from src.EvolutionaryAlgorithm import EvolutionaryAlgorithm as EA
from src.operations.selection.Tournament import Tournament
from src.operations.crossover.Order import Order
from src.operations.mutation.Exchange import Exchange
from src.Population import Population

class _MinTournament:
    def __init__(self, k: int = 3, rng: random.Random | None = None):
        self.k = k
        self.rng = rng or random.Random()

    def __call__(self, population: Population, num_to_select: int) -> Population:
        n_nodes = len(population.individuals[0].permutation)
        tsp = population.individuals[0].tsp
        winners = Population(population_size=num_to_select, number_of_nodes=n_nodes, tsp=tsp)

        inds = population.individuals
        k = min(self.k, len(inds))
        for _ in range(num_to_select):
            competitors = self.rng.sample(inds, k)
            winners.individuals.append(min(competitors, key=lambda ind: ind.fitness))  # MIN
        return winners

class EvolutionaryAlgorithm(EA):
    ### Variant A: Generational GA with elitism, configurable parent selection
    ### OX crossover, and swap mutation

    def __init__(
        self,
        filepath: str,
        population_size: int = 100,
        distance_metric: str = "euclidean",
        precompute_distances: bool = True,
        # operators
        selection: Optional[object] = None,
        crossover: Optional[object] = None,
        mutation: Optional[object] = None,
        # hyperparameters
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.2,
        elitism_k: int = 2,
        seed: Optional[int] = None,
        log_dir: str = "results/ea_variant_a",
    ):
        # Prepare a local RNG so we can seed Order() before base init
        rng_local = random.Random(seed)

        # base problem
        super().__init__(
            filepath=filepath,
            distance_metric=distance_metric,
            precompute_distances=precompute_distances,
            population_size=population_size,
            selection=None,  # set after super() so self.population exists
            crossover=crossover or Order(rng=rng_local),
            mutation=mutation or Exchange(),
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elitism_k=elitism_k,
            seed=seed,
            log_dir=log_dir,
        )

        # RNG
        self.seed = seed
        self.rng = rng_local
        np.random.seed(seed if seed is not None else None)

        # operators
        self.selection = selection or _MinTournament(k=3, rng=self.rng)
        self.crossover = self._crossover_ops[0] if self._crossover_ops else (crossover or Order(rng=self.rng))
        self.mutation = self.mutation or Exchange()

        # hyperparams
        self.crossover_rate = float(crossover_rate)
        self.mutation_rate = float(mutation_rate)
        self.elitism_k = int(elitism_k)

        # logging
        # (Handled by the base EA; keeping section name for consistency)

        # initial population fitness computation test
        for ind in self.population.individuals:
            if ind.fitness is None:
                ind.calculate_fitness()

    def _elitism(self, pop: Population, k: int) -> list:
        # delegate to base but keep the same function name/signature
        return super()._elitism(pop, k)

    def _select_parents(self, pop: Population, n_pairs: int) -> list:
        #returns list[tuple[Individual, Individual]]
        #pick 2 per pair (with replacement)
        return super()._select_parents(pop, n_pairs)

    def _maybe_crossover(self, p1: Individual, p2: Individual) -> tuple:
        #returns tuple[Individual, Individual]
        if self.rng.random() < self.crossover_rate and self.crossover is not None:
            return self.crossover.xover(p1, p2)
        # clone parents
        c1 = self._clone(p1)
        c2 = self._clone(p2)
        return (c1, c2)

    def _maybe_mutate(self, child: Individual) -> Individual:
        if self.rng.random() < self.mutation_rate and self.mutation is not None:
            n = len(child.permutation)
            i, j = self.rng.sample(range(n), 2)
            child = self.mutation.mutate_individual(child, i, j, update_fitness=True)
        return child

    def _log_arrays(self, instance_name: str, seed: Optional[int], best_hist: list[float], mean_hist: list[float]):
        # delegate to base to actually write files
        return super()._log_arrays(instance_name, seed, best_hist, mean_hist)

    ### public API ###

    def solve(self, max_generations: int = 2000) -> Individual:
        """
        Runs Variant A and returns the best individual found.
        """
        # Use the generic EA loop (elitism -> selection -> crossover -> mutation)
        return super().solve(max_generations=max_generations)