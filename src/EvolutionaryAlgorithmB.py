from __future__ import annotations

import numpy as np
import random
from typing import Optional

from src.EvolutionaryAlgorithm import EvolutionaryAlgorithm as EA
from src.Population import Population
from src.Individual import Individual

# Importing crossover operators 
from src.operations.selection.FitnessBased import FitnessBased
from src.operations.crossover.PMX import PMX
from src.operations.crossover.Cycle import Cycle


class EvolutionaryAlgorithm(EA):

    # Variant B: Generational GA with elitism, Crossover Only

    def __init__(
        self,
        filepath: str,
        population_size: int = 100,
        distance_metric: str = "euclidean",
        precompute_distances: bool = True,
        # operators
        selection: Optional[object] = None,
        crossover1: Optional[object] = None,
        crossover2: Optional[object] = None, # No mutations 
        # hyperparameters
        crossover_rate: float = 0.65,
        elitism_k: int = 2,
        seed: Optional[int] = None,
        log_dir: str = "results/ea_variant_b",
    ):
        # base problem
        super().__init__(
        filepath=filepath,
        population_size=population_size,
        distance_metric=distance_metric,
        precompute_distances=precompute_distances,
        selection=None,  # set after super()
        crossover=(crossover1 or PMX(), crossover2 or Cycle()),  # tuple => base class handles choice
        mutation=None,                       # Since Variant B is crossover-only
        crossover_rate=crossover_rate,
        mutation_rate=0.0,
        elitism_k=elitism_k,
        seed=seed,
        log_dir=log_dir,
        )
        self.selection = selection or FitnessBased()


        # RNG Setup
        self.seed = seed
        self.rng = random.Random(seed)
        np.random.seed(seed if seed is not None else None)

        # operators
        self.selection  = selection  or FitnessBased()
        self.crossover1 = crossover1 or PMX()
        self.crossover2 = crossover2 or Cycle()

        # hyperparameters 
        self.crossover_rate = float(crossover_rate)
        self.elitism_k      = int(elitism_k)

        # logging
        # (Handled by the base class)

        # Calculate initial fitness values
        for ind in self.population.individuals:
            if ind.fitness is None:
                ind.calculate_fitness()

    def _maybe_crossover(self, p1, p2):
        if not isinstance(p1.permutation, np.ndarray):
            p1.permutation = np.array(p1.permutation, dtype=int)
        if not isinstance(p2.permutation, np.ndarray):
            p2.permutation = np.array(p2.permutation, dtype=int)
        return super()._maybe_crossover(p1, p2)

    # public API 
    def solve(self, max_generations: int = 2000) -> Individual:
        # Runs Variant B and returns the best individual found.
        
        # Calls back to the generic EA loop (elitism + selection => crossover)
        return super().solve(max_generations=max_generations)
