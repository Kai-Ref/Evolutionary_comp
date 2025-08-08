from src.TSP import TSP
import numpy as np
from src.Individual import Individual

class EvolutionaryAlgorithm(TSP):
    def __init__(self, population_size: int, node_coords: np.ndarray, distance_metric: str = 'euclidean', precompute_distances: bool = True, mutation=None, crossover=None, selection=None):
        super().__init__(population_size, node_coords, distance_metric, precompute_distances)
        self.population = [Individual(self.node_coords.shape[0]) for _ in range(population_size)]
        self.mutation = mutation
        self.crossover = crossover
        self.selection = selection
    
    def solve(self):
        # Implement the evolutionary algorithm here
        raise NotImplementedError("Evolutionary algorithm is not implemented yet.")
    