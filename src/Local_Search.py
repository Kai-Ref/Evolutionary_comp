import numpy as np
from typing import override
from src.TSP import TSP
from src.Population import Population
from src.Individual import Individual
from typing import Generator
class LocalSearch(TSP):
    def __init__(self, filepath: str, distance_metric: str = 'euclidean', precompute_distances: bool = True, mutation=None, population_size: int = 1):
        super().__init__(filepath=filepath, distance_metric=distance_metric, precompute_distances=precompute_distances, population_size=population_size, mutation=mutation)
        
    @override
    def solve(self, max_iterations: int = 1E4) -> None:
        for individual_index in range(len(self.population)):    
            for _ in range(max_iterations):
                new_individual = self.perform_one_step(self.population[individual_index].copy())
                if new_individual is None:
                    break
                self.population[individual_index] = new_individual

    def perform_one_step(self, current: Individual) -> Individual | None:
        for neighbour in self.get_next_neighbour(current):
            if neighbour.fitness > current.fitness:
                return neighbour
        return None

    def get_next_neighbour(self, current: Individual) -> Generator[Individual, None, None]:
        """
        Generator Function which returns the next neighbour. Systematically goes through all possible mutations. An unique mutation is characterized by (i, j) indices.
        It returns None if all possible neighbours where already returned
        """
        indices = [(i, j) for i in range(self.node_coords.shape[0]) for j in range(self.node_coords.shape[0]) if i != j]
        rng = np.random.default_rng()
        rng.shuffle(indices)
        for i, j in indices:
            yield self.mutation(current, i, j)