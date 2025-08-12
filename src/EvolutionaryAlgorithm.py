from src.TSP import TSP
import numpy as np
from src.Individual import Individual

class EvolutionaryAlgorithm(TSP):
    def __init__(self, filepath: str,population_size: int, distance_metric: str = 'euclidean', precompute_distances: bool = True, mutation=None, crossover=None, selection=None):
        super().__init__(filepath=filepath, distance_metric=distance_metric, precompute_distances=precompute_distances, population_size=population_size, mutation=mutation)
        self.population = [Individual(self.node_coords.shape[0]) for _ in range(population_size)]
        self.crossover = crossover
        self.selection = selection
    
    def solve(self):
        # Implement the evolutionary algorithm here
        raise NotImplementedError("Evolutionary algorithm is not implemented yet.")
    