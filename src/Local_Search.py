import numpy as np
from typing import override
from src.TSP import TSP
from src.Population import Population
from src.Individual import Individual
from typing import Generator
from tqdm import tqdm
import time
import os

class LocalSearch(TSP):
    """
    Local Search solver for the Traveling Salesman Problem.

    Inherits from TSP and performs local search using neighborhood mutation operations
    (e.g., jump, exchange, 2-opt) to iteratively improve solutions.

    Attributes:
        fitness_of_finished_individuals (list): History of fitness values for completed individuals.
        indices (list): List of all unique (i, j) index pairs for neighborhood operations.
        number_neighbors (int): Number of neighbors to explore per iteration.
    """
    def __init__(self, 
                filepath: str, 
                distance_metric: str = 'euclidean', 
                precompute_distances: bool = True, 
                mutation=None, 
                population_size: int = 1):
        """
        Initialize LocalSearch, the superclass TSP instance and local search parameters (Indices and number of neighbors).

        Args:
            filepath (str): Path to the TSP dataset file.
            distance_metric (str): Distance metric to use ('euclidean' by default).
            precompute_distances (bool): Whether to precompute the distance matrix to save computational complexity creep in iterations.
            mutation: Mutation operator for generating neighbors.
            population_size (int): Number of individuals in the population.
        """

        # First initialize the TSP superclass object
        super().__init__(filepath=filepath, 
                        distance_metric=distance_metric, 
                        precompute_distances=precompute_distances, 
                        population_size=population_size, 
                        mutation=mutation)

        self.fitness_of_finished_individuals=[]

        # Create all indice combination once in advance to reduce computational load
        self.indices = ((i, j) for i in range(self.node_coords.shape[0]) for j in range(self.node_coords.shape[0]) if i != j)
        self.indices = list(self.indices)

    @override
    def solve(self, max_iterations: int = 1E4) -> None:
        """
        Run the local search on the population.

        Iteratively improves each individual by exploring neighbors until
        a stopping criterion is met.
        
        The stopping criteria implemented here is either:
        - the maximum number of iterations (max_iterations) is reached, 
        - less than 1% improvement over the last 10 iterations, 
        - or no better neighbor exists.

        After each iteration, the current fitness value is added to the fitness_history.
        When a stopping criterion is met, the list is saved.

        Args:
            max_iterations (int): Maximum number of iterations per individual.
        """
        for individual_index in range(len(self.population)):
            fitness_history = [self.population[individual_index].fitness]
            pbar = tqdm(range(max_iterations), desc=f"Ind {individual_index} Fitness: {self.population[individual_index].fitness:.2f}")    
            for iteration in pbar:
                # Perform one optimized local search step
                new_individual = self.perform_one_step(self.population[individual_index].copy())
                
                # Stop if no improvement found (local optimum)
                if new_individual is None:
                    print(f"Individual {individual_index} reached local optimum, at iteration {iteration}")
                    break
                
                fitness_history.append(new_individual.fitness)
                self.population[individual_index] = new_individual
                pbar.set_description(f"Ind {individual_index} Fitness: {new_individual.fitness:.2f}")

                # Stopping criterion: <1% improvement over last 10 steps
                recent_fitness = fitness_history[-10:]  # Get the last 10 fitness values
                if len(recent_fitness) == 10:
                    initial_fitness = recent_fitness[0]
                    final_fitness = recent_fitness[-1]
                    if initial_fitness != 0:  # avoid division by zero
                        improvement = (initial_fitness - final_fitness) / abs(initial_fitness)
                        if improvement < 0.01:
                            print(f"Stopping early: <1% improvement over last 10 iterations at iteration {iteration}")
                            break
            
            # Save fitness history
            path_prefix = self.filepath.removeprefix("datasets/").rsplit('.', 1)[0]
            run_number = self.get_next_run_number(path_prefix, self.mutation)
            self.file_writer(
                fitness_history,
                name=f"{path_prefix}/{self.mutation}/Individual_{run_number}_fitness_history_individual"
            )
    

    def get_next_run_number(self, path_prefix, mutation):
        """
        Determine the next run number for saving fitness history.

        Args:
            path_prefix (str): Prefix path for the dataset.
            mutation: Mutation type used for the run.

        Returns:
            int: Next available run number.
        """
        folder = f"data/{path_prefix}/{mutation}"
        os.makedirs(folder, exist_ok=True)  # create folder if it doesn't exist
        existing_files = os.listdir(folder)
        run_numbers = []

        for f in existing_files:
            if "Individual_" in f:
                try:
                    num = int(f.split("Individual_")[1].split("_")[0])
                    run_numbers.append(num)
                except:
                    continue

        return max(run_numbers) + 1 if run_numbers else 1
        
    def perform_one_step(self, current: Individual) -> Individual | None:
        """
        Perform one local search step.

        Args:
            current (Individual): Current individual to improve.

        Returns:
            Individual or None: The improved individual or None if no improvement.
        """
        # 1. Retrieve the best possible neighbor 
        best_pair, best_fitness = self.get_best_neighbor(current)

        # 2. If there is a better neighbor, return it
        if best_pair and best_fitness < current.fitness:
            i, j = best_pair
            # mutate the individual, but do not update the fitness immediately, since we already calculated it
            new_individual = self.mutation.mutate_individual(current, i, j, update_fitness=False)
            new_individual.fitness = best_fitness
            return new_individual
        return None

    def get_best_neighbor(self, current: Individual):
        """
        Find the best neighbor using vectorized fitness calculation, based only on those edges that were changed.

        Args:
            current (Individual): Current individual.

        Returns:
            tuple: (best index pair (i, j), best fitness)
        """
        best_delta = 0
        best_pair = None

        # performs vectorized fitness calculation
        deltas = self.mutation.efficient_fitness_calculation_vectorized(current, self.indices)

        # select the best neighbor and return it with its fitness 
        best_idx = np.argmin(deltas)
        best_delta = deltas[best_idx]
        best_pair = tuple(self.indices[best_idx])
        return best_pair, current.fitness + best_delta
