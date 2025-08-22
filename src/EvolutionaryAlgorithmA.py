from src.TSP import TSP
from src.Population import Population
from src.Individual import Individual
from src.FileWriter import FileWriter

from src.operations.selection.Tournament import Tournament
from src.operations.selection.FitnessBased import FitnessBased
from src.operations.crossover.Order import Order
from src.operations.mutation.Exchange import Exchange

import numpy as np
import random
import os
from typing import Optional


class EvolutionaryAlgorithm(TSP):

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
        # base problem
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

        # operators
        self.selection = selection or Tournament(k=3, rng=self.rng)
        self.crossover = crossover or Order(rng=self.rng)
        self.mutation = mutation or Exchange()

        # hyperparams
        self.crossover_rate = float(crossover_rate)
        self.mutation_rate = float(mutation_rate)
        self.elitism_k = int(elitism_k)

        # logging
        self.file_writer = FileWriter()
        self.log_dir_base = log_dir

        # initial population fitness computation test
        for ind in self.population.individuals:
            if ind.fitness is None:
                ind.calculate_fitness()


    def _elitism(self, pop: Population, k: int) -> list[Individual]:
        return sorted(pop.individuals, key=lambda ind: ind.fitness)[:max(0, k)]

    def _select_parents(self, pop: Population, n_pairs: int) -> list[tuple[Individual, Individual]]:
        #pick 2 per pair (with replacement)
        pairs = []
        for _ in range(n_pairs):
            sel = self.selection(pop, 2)
            p1, p2 = sel.individuals[0], sel.individuals[1]
            pairs.append((p1, p2))
        return pairs

    def _maybe_crossover(self, p1: Individual, p2: Individual) -> tuple[Individual, Individual]:
        if self.rng.random() < self.crossover_rate:
            return self.crossover.xover(p1, p2)
        # clone parents
        c1 = Individual(number_of_nodes=None, tsp=p1.tsp, permutation=p1.permutation.copy())
        c1.fitness = p1.fitness
        c2 = Individual(number_of_nodes=None, tsp=p2.tsp, permutation=p2.permutation.copy())
        c2.fitness = p2.fitness
        return c1, c2

    def _maybe_mutate(self, child: Individual) -> Individual:
        if self.rng.random() < self.mutation_rate:
            n = len(child.permutation)
            i, j = self.rng.sample(range(n), 2)
            child = self.mutation.mutate_individual(child, i, j, update_fitness=True)
        return child

    def _log_arrays(self, instance_name: str, seed: Optional[int], best_hist: list[float], mean_hist: list[float]):
        seed_folder = f"seed_{seed}" if seed is not None else "seed_none"
        folder = os.path.join(self.log_dir_base, instance_name, seed_folder)
        os.makedirs(folder, exist_ok=True)

        self.file_writer.file_path = folder
        self.file_writer(np.array(best_hist), "best_cost_per_generation.npy")
        self.file_writer(np.array(mean_hist), "mean_cost_per_generation.npy")

    ### public API ###

    def solve(self, max_generations: int = 2000) -> Individual:
        """
        Runs Variant A and returns the best individual found.
        """
        pop = self.population  # Population object
        n = len(pop.individuals)
        instance_name = os.path.splitext(os.path.basename(self.filepath))[0]

        best_history = []
        mean_history = []

        # initial stats
        best = min(pop.individuals, key=lambda ind: ind.fitness)
        best_history.append(best.fitness)
        mean_history.append(sum(ind.fitness for ind in pop.individuals) / n)

        # generational loop
        for _ in range(int(max_generations)):
            elites = self._elitism(pop, self.elitism_k)

            # creating children
            n_pairs = (n - len(elites)) // 2
            next_inds: list[Individual] = elites.copy()

            for (p1, p2) in self._select_parents(pop, n_pairs):
                c1, c2 = self._maybe_crossover(p1, p2)
                c1 = self._maybe_mutate(c1)
                c2 = self._maybe_mutate(c2)
                next_inds.extend([c1, c2])

            # if odd top up using selection
            while len(next_inds) < n:
                sel = self.selection(pop, 1)
                lone = sel.individuals[0]
                clone = Individual(number_of_nodes=None, tsp=lone.tsp, permutation=lone.permutation.copy())
                clone.fitness = lone.fitness
                clone = self._maybe_mutate(clone)
                next_inds.append(clone)

            # swap populations
            pop.individuals = next_inds[:n]
            pop.population_size = n

            # stats
            cur_best = min(pop.individuals, key=lambda ind: ind.fitness)
            best = cur_best if cur_best.fitness < best.fitness else best

            best_history.append(best.fitness)
            mean_history.append(sum(ind.fitness for ind in pop.individuals) / n)

        # logs
        self._log_arrays(instance_name, self.seed, best_history, mean_history)
        return best
