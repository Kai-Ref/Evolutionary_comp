import numpy as np
from typing import override
from src.TSP import TSP
from src.Population import Population
from src.Individual import Individual
from typing import Generator
from tqdm import tqdm
import time

class LocalSearch(TSP):
    def __init__(self, filepath: str, distance_metric: str = 'euclidean', precompute_distances: bool = True, mutation=None, population_size: int = 1, number_neighbors:int = 1, max_neighbours:int = -1):
        super().__init__(filepath=filepath, distance_metric=distance_metric, precompute_distances=precompute_distances, population_size=population_size, mutation=mutation)
        self.fitness_of_finished_individuals=[]
        # if max_neighbours > 0:
        #     self.max_neighbours = max_neighbours
        # else:
        #     self.max_neighbours = self.node_coords.shape[0] * (self.node_coords.shape[0] - 1) // 2  # All unique pairs of nodes
        # self.max_neighbours = max(1,int(self.max_neighbours/self.number_neighbors))

        # moved this into the init, since the indices are always the same
        self.indices = ((i, j) for i in range(self.node_coords.shape[0]) for j in range(self.node_coords.shape[0]) if i != j)
        self.indices = list(self.indices)
        if number_neighbors > 0:
            if self.number_neighbors > len(self.indices):
                raise ValueError(f"Number of neighbors {self.number_neighbors} exceeds the number of unique pairs {len(self.indices)}.")
            self.number_neighbors = number_neighbors
        else:
            self.number_neighbors = len(self.indices)  # Use all unique pairs if number_neighbors is not set or <= 0
    
    @override
    def solve(self, max_iterations: int = 1E4) -> None:
        for individual_index in range(len(self.population)):
            fitness_history = np.array([self.population[individual_index].fitness])
            pbar = tqdm(range(max_iterations), desc=f"Ind {individual_index} Fitness: {self.population[individual_index].fitness:.2f}")    
            for iteration in pbar:
                # for _ in range(self.max_neighbours):
                # start =  time.time() 
                # new_individual = self.perform_one_step(self.population[individual_index].copy())
                # end = time.time() 
                # print(f"Old Elapsed time: {end - start:.6f} seconds with Individual {new_individual}")
                # start =  time.time()
                new_individual = self.perform_one_step_optimized(self.population[individual_index].copy())
                # end = time.time() 
                # print(f"New Elapsed time: {end - start:.6f} seconds with Individual {new_individual}")
                if new_individual is None:
                    print(f"Individual {individual_index} reached local optimum, at iteration {iteration}")
                    break
                fitness_history = np.append(fitness_history, new_individual.fitness)
                self.population[individual_index] = new_individual
                pbar.set_description(f"Ind {individual_index} Fitness: {new_individual.fitness:.2f}")
            self.fitness_of_finished_individuals.append(fitness_history)
            self.file_writer(fitness_history, name=f"fitness_history_individual")
        print(self.fitness_of_finished_individuals)
        
    def perform_one_step_optimized(self, current: Individual) -> Individual | None:
        best_pair, best_fitness = self.get_best_neighbour_delta(current)
        if best_pair and best_fitness < current.fitness:
            i, j = best_pair
            # mutate the individual, but do not update the fitness immediately, since we already calculated it
            new_individual = self.mutation.mutate_individual(current, i, j, update_fitness=False)
            new_individual.fitness = best_fitness 
            return new_individual
        return None
    
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
        
    def shuffle_neighbour_indices(self) -> Generator[tuple[int, int], None, None]:
        """
        Generator that yields all unique (i, j) index pairs for neighbours, shuffled.
        """
        rng = np.random.default_rng()
        rng.shuffle(self.indices)
        for idx_pair in self.indices:
            yield idx_pair
        
    def get_next_neighbour(self, current: Individual) -> Generator[Individual, None, None]:
        """
        Generator Function which returns the next neighbour. Systematically goes through all possible mutations. An unique mutation is characterized by (i, j) indices.
        It returns None if all possible neighbours where already returned
        """
        # only shuffle indices if we select a neighbourhood subset, since the order is irrelevant when considering all indices
        if len(self.indices) > self.number_neighbors:
            indices = list(self.shuffle_neighbour_indices())
        else:
            indices = self.indices

        for i, j in indices:
            yield self.mutation.mutate_individual(current, i, j, update_fitness=True)

    def get_best_neighbour_delta(self, current: Individual):
        best_delta = 0
        best_pair = None
        for i, j in self.indices:
            delta = self.mutation.efficient_fitness_calculation(current, i, j)
            if delta < best_delta:
                best_delta = delta
                best_pair = (i, j)
        return best_pair, current.fitness + best_delta
