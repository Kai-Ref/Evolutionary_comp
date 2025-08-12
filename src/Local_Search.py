import numpy as np
from typing import override
from src.TSP import TSP
class LocalSearch(TSP):
    def __init__(self, filepath: str, distance_metric: str = 'euclidean', precompute_distances: bool = True, mutation=None, population_size: int = 1):
        super().__init__(filepath=filepath, distance_metric=distance_metric, precompute_distances=precompute_distances, population_size=population_size, mutation=mutation)
        
    @override
    def solve(self):
        
    

