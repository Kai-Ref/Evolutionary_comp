import numpy as np
from typing import override
from src.TSP import TSP
from src.Population import Population
class LocalSearch(TSP):
    def __init__(self, filepath: str, distance_metric: str = 'euclidean', precompute_distances: bool = True, mutation=None, population_size: int = 1):
        super().__init__(filepath=filepath, distance_metric=distance_metric, precompute_distances=precompute_distances, population_size=population_size, mutation=mutation)
        
    @override
    def solve(self, max_iterations: int = 1E4):
        
    
    def get_next_neighbour(self, current: Population) -> Population:
        """
        Generator Function which returns the next neighbour. It returns None if all possible neighbours where already returned
        """
        neighbour = 

    
