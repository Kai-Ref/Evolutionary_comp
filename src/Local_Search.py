import numpy as np
from typing import override
from src.TSP import TSP
from src.Population import Population
from src.Individual import Individual
from typing import Generator
from tqdm import tqdm
class LocalSearch(TSP):
    def __init__(self, filepath: str, distance_metric: str = 'euclidean', precompute_distances: bool = True, mutation=None, population_size: int = 1, number_neighbors:int = 1):
        super().__init__(filepath=filepath, distance_metric=distance_metric, precompute_distances=precompute_distances, population_size=population_size, mutation=mutation)
        self.previous_fitness = np.expand_dims(np.array([individuals.fitness for individuals in self.population]), axis=0)
        self.number_neighbors = number_neighbors

    @override
    def solve(self, max_iterations: int = 1E4) -> None:
        for individual_index in range(len(self.population)):
            pbar = tqdm(range(max_iterations), desc=f"Ind {individual_index} Fitness: {self.population[individual_index].fitness:.2f}")    
            for iteration in pbar:
                new_individual = self.perform_one_step(self.population[individual_index].copy())
                if new_individual is None:
                    print(f"Individual {individual_index} reached local optimum, at iteration {iteration}")
                    break
                self.population[individual_index] = new_individual
                pbar.set_description(f"Ind {individual_index} Fitness: {new_individual.fitness:.2f}")
            self.previous_fitness = np.expand_dims(np.array([individuals.fitness for individuals in self.population]), axis=0)

        
    def perform_one_step(self, current: Individual) -> Individual | None:
        neighbours = []
        for idx, neighbour in zip(range(self.number_neighbors), self.get_next_neighbour(current)):
            # neighbour.calculate_fitness() # Since we return the indivduals now, instead of overwriting, we can omit this
            neighbours.append(neighbour)

        # Find the fittest neighbour and return it if it improves the fitness
        best_neighbour = min(neighbours, key=lambda individual: individual.fitness)
        # print(f"Current fitness {current.fitness}, best neighbour fitness {best_neighbour.fitness}")
        if best_neighbour.fitness < current.fitness:
            return best_neighbour
        else:
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
            yield self.mutation.mutate_individual(current, i, j)