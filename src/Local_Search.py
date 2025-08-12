import numpy as np
from typing import override
from src.TSP import TSP
from src.Population import Population
from src.Individual import Individual
from typing import Generator
class LocalSearch(TSP):
    def __init__(self, filepath: str, distance_metric: str = 'euclidean', precompute_distances: bool = True, mutation=None, population_size: int = 1):
        super().__init__(filepath=filepath, distance_metric=distance_metric, precompute_distances=precompute_distances, population_size=population_size, mutation=mutation)
        self.previous_fitness = np.array([individuals.fitness for individuals in self.population]).expand_dims(axis=0)

    @override
    def solve(self, max_iterations: int = 1E4) -> None:
        individuals_reached_optimum = 0
        for iteration in range(max_iterations):
            for individual_index in range(len(self.population)):    
                new_individual = self.perform_one_step(self.population[individual_index].copy()) if self.population[individual_index].fitness is not None else None
                if new_individual is None:
                    self.population[individual_index].is_local_optimum = True
                    individuals_reached_optimum += 1
                    print(f"Individual {individual_index} reached local optimum, at iteration {iteration}.\n It is the {individuals_reached_optimum}th individual(out of {len(self.population)}) to reach local optimum.")
                    break
                self.population[individual_index] = new_individual
            self.previous_fitness = np.array([individuals.fitness for individuals in self.population]).expand_dims(axis=0)
            if individuals_reached_optimum == len(self.population):
                print(f"All individuals reached local optimum at iteration {iteration}.")
                break
        
    def perform_one_step(self, current: Individual) -> Individual | None:
        for neighbour in self.get_next_neighbour(current):
            if neighbour.fitness is None:
                neighbour.calculate_fitness()
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
            yield self.mutation.mutate(current, i, j)