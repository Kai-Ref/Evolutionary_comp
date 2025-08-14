import numpy as np
from typing import override
from src.TSP import TSP
from src.Population import Population
from src.Individual import Individual
from typing import Generator
from tqdm import tqdm
class LocalSearch(TSP):
    def __init__(self, filepath: str, distance_metric: str = 'euclidean', precompute_distances: bool = True, mutation=None, population_size: int = 1, number_neighbors:int = 1, max_neighbours:int = -1):
        super().__init__(filepath=filepath, distance_metric=distance_metric, precompute_distances=precompute_distances, population_size=population_size, mutation=mutation)
        self.fitness_of_finished_individuals=[]
        self.number_neighbors = number_neighbors
        if max_neighbours > 0:
            self.max_neighbours = max_neighbours
        else:
            self.max_neighbours = self.node_coords.shape[0] * (self.node_coords.shape[0] - 1) // 2  # All unique pairs of nodes
        self.max_neighbours = max(1,int(self.max_neighbours/self.number_neighbors))
    @override
    def solve(self, max_iterations: int = 1E4) -> None:
        for individual_index in range(len(self.population)):
            fitness_history = np.array([self.population[individual_index].fitness])
            pbar = tqdm(range(max_iterations), desc=f"Ind {individual_index} Fitness: {self.population[individual_index].fitness:.2f}")    
            for iteration in pbar:
                for _ in range(self.max_neighbours):
                    new_individual = self.perform_one_step(self.population[individual_index].copy())
                    if new_individual is not None:
                        break
                if new_individual is None:
                    print(f"Individual {individual_index} reached local optimum, at iteration {iteration}")
                    break
                fitness_history = np.append(fitness_history, new_individual.fitness)
                self.population[individual_index] = new_individual
                pbar.set_description(f"Ind {individual_index} Fitness: {new_individual.fitness:.2f}")
            self.fitness_of_finished_individuals.append(fitness_history)
            self.file_writer(fitness_history, name=f"fitness_history_individual")
        print(self.fitness_of_finished_individuals)
        
    def perform_one_step(self, current: Individual) -> Individual | None:
        neighbours = []
        for idx, neighbour in zip(range(self.number_neighbors), self.get_next_neighbour(current)):
            # neighbour.calculate_fitness() # Since we return the indivduals now, instead of overwriting, we can omit this
            neighbours.append(neighbour) if neighbour.fitness < current.fitness else None

        # Find the fittest neighbour and return it if it improves the fitness
        best_neighbour = min(neighbours, key=lambda individual: individual.fitness) if neighbours else current
        # print(f"Current fitness {current.fitness}, best neighbour fitness {best_neighbour.fitness}")
        if best_neighbour.fitness < current.fitness:
            return best_neighbour
        else:
            return None
    def neighbour_indices(self) -> Generator[tuple[int, int], None, None]:
        """
        Generator that yields all unique (i, j) index pairs for neighbours, shuffled.
        """
        indices = ((i, j) for i in range(self.node_coords.shape[0]) for j in range(self.node_coords.shape[0]) if i != j)
        indices = list(indices)
        rng = np.random.default_rng()
        rng.shuffle(indices)
        for idx_pair in indices:
            yield idx_pair
        
    def get_next_neighbour(self, current: Individual) -> Generator[Individual, None, None]:
        """
        Generator Function which returns the next neighbour. Systematically goes through all possible mutations. An unique mutation is characterized by (i, j) indices.
        It returns None if all possible neighbours where already returned
        """
        indices = list(self.neighbour_indices())
        for i, j in indices:
            yield self.mutation.mutate_individual(current, i, j, update_fitness=True)