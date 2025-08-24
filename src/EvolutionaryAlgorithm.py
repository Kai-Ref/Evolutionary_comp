from __future__ import annotations

import os
import random
from typing import Optional, List, Tuple, Iterable, Sequence, Union

import numpy as np

from src.TSP import TSP
from src.Population import Population
from src.Individual import Individual
from src.FileWriter import FileWriter


class EvolutionaryAlgorithm(TSP):
    """
    Evolutionary Alogithm class to be used by all the different variants

    Operators: 
    Selection
    Crossover
    Mutation

    Hyperparams:
    crossover_rate: probability of applying crossover (otherwise clones parents)
    mutation_rate: probability of mutating each child (variants can override `_maybe_mutate`)
    elitism_k: number of elite solutions copied to next generation
    """

    def __init__(
        self,
        filepath: str,
        population_size: int,
        distance_metric: str = "euclidean",
        precompute_distances: bool = True,
        # operators
        selection: Optional[object] = None,
        crossover: Optional[Union[object, Sequence[object]]] = None,
        mutation: Optional[object] = None,
        # hyperparameters
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.2,
        elitism_k: int = 2,
        seed: Optional[int] = None,
        log_dir: str = "results/ea_generic",
    ):
        super().__init__(
            filepath=filepath,
            distance_metric=distance_metric,
            precompute_distances=precompute_distances,
            population_size=population_size,
            mutation=None,  # Mutaations are handled by ourselves 
        )

        # RNG Setup
        self.seed = seed
        self.rng = random.Random(seed)
        if seed is not None:
            np.random.seed(seed)

        # Storage Operators
        self.selection = selection  # variants may set a default
        if crossover is None:
            self._crossover_ops: List[object] = []
        elif isinstance(crossover, (list, tuple)):
            self._crossover_ops = list(crossover)
            # allow random.choice on list-like:
            self._crossover_ops_choice = self._crossover_ops
        else:
            self._crossover_ops = [crossover]
            self._crossover_ops_choice = self._crossover_ops

        self.mutation = mutation

        # Hyperparameters 
        self.crossover_rate = float(crossover_rate)
        self.mutation_rate = float(mutation_rate)
        self.elitism_k = int(elitism_k)

        # Logging Operators
        self.file_writer = FileWriter()
        self.log_dir_base = log_dir

        # Calculate initial fitness values
        for individual in self.population.individuals:
            if individual.fitness is None:
                individual.calculate_fitness()

    # Helper Functions
    def _elitism(self, population: Population, k: int) -> List[Individual]:
        # Return K best individuals from the population
        return sorted(population.individuals, key=lambda ind: ind.fitness)[:max(0, k)]

    def _select_parents(self, population: Population, n_pairs: int) -> List[Tuple[Individual, Individual]]:
        # Selects parent pair to reproduce 
        if not self.selection:
            raise RuntimeError("No selection operator provided.")
        pairs: List[Tuple[Individual, Individual]] = []
        for _ in range(n_pairs):
            parents = self.selection(population, 2)
            pairs.append((parents.individuals[0], parents.individuals[1]))
        return pairs

    def _clone(self, individual: Individual) -> Individual:
        clone = Individual(
            number_of_nodes=None
            tsp=individual.tsp
            permutation=individual.permutation.copy()
            )
        clone.fitness = individual.fitness
        return clone

    def _maybe_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        # Apply crossover to create offspring, or clone parents if no crossover occurs
        if not self._crossover_ops or self.rng.random() >= self.crossover_rate:
            child1, child2 = self._clone(parent1), self._clone(parent2)
        else:
            op = self.rng.choice(self._crossover_ops) if len(self._crossover_ops) > 1 else self._crossover_ops[0]
            child1, child2 = op.xover(parent1, parent2)

        if child1.fitness is None:
            child1.calculate_fitness()
        if child2.fitness is None:
            child2.calculate_fitness()
        return child1, child2

    def _maybe_mutate(self, child: Individual) -> Individual:
        # Apply mutation to an individual based on mutation rate.
        if self.mutation is None or self.mutation_rate <= 0.0:
            return child
        if self.rng.random() < self.mutation_rate:
            n = len(child.permutation)
            i, j = self.rng.sample(range(n), 2)
            child = self.mutation.mutate_individual(child, i, j, update_fitness=True)
        return child

    def _log_arrays(self, instance_name: str, seed: Optional[int], best_hist: List[float], mean_hist: List[float]):
        # Logs the evolution history to .npy files
        seed_folder = f"seed_{seed}" if seed is not None else "seed_none"
        folder = os.path.join(self.log_dir_base, instance_name, seed_folder)
        os.makedirs(folder, exist_ok=True)

        self.file_writer.file_path = folder
        self.file_writer(np.array(best_hist), f"{instance_name}_best_cost_per_generation.npy")
        self.file_writer(np.array(mean_hist), f"{instance_name}_mean_cost_per_generation.npy")

    # --------- generic solve (variants may override if they need extras) ---------

    def solve(self, max_generations: int = 2000) -> Individual:
        """
        Generic G loop: elitism + (selection -> crossover -> mutation).
        Variants override to insert extra steps (e.g. VariantC(Mutations Only) 
        employs memetic polish, immigrants).
        """
        pop = self.population
        n = len(pop.individuals)
        instance_name = os.path.splitext(os.path.basename(self.filepath))[0]

        best_history: List[float] = []
        mean_history: List[float] = []

        best = min(pop.individuals, key=lambda ind: ind.fitness)
        best_history.append(best.fitness)
        mean_history.append(sum(ind.fitness for ind in pop.individuals) / n)

        for gen in range(int(max_generations)):
            elites = self._elitism(pop, self.elitism_k)

            n_pairs = (n - len(elites)) // 2
            next_inds: List[Individual] = elites.copy()

            for (parent1, parent2) in self._select_parents(pop, n_pairs):
                child1, child2 = self._maybe_crossover(parent1, parent2)
                child1 = self._maybe_mutate(child1)
                child2 = self._maybe_mutate(child2)
                next_inds.extend([child1, child2])

            # top-up if odd
            while len(next_inds) < n:
                lone = self.selection(pop, 1).individuals[0]
                clone = self._maybe_mutate(self._clone(lone))
                next_inds.append(clone)

            pop.individuals = next_inds[:n]
            pop.population_size = n

            cur_best = min(pop.individuals, key=lambda ind: ind.fitness)
            if cur_best.fitness < best.fitness:
                best = cur_best

            best_history.append(best.fitness)
            mean_history.append(sum(ind.fitness for ind in pop.individuals) / n)

            if gen in [2000, 5000, 10000, 20000]:
                try:
                    print(f"{gen} mean: {best.fitness}")
                except Exception:
                    pass

        self._log_arrays(instance_name, self.seed, best_history, mean_history)
        return best
