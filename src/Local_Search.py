import numpy as np
from typing import override
from src.TSP import TSP
class LocalSearch(TSP):
    def __init__(self, population_size: int, node_coords: np.ndarray, distance_metric: str = 'euclidean', precompute_distances: bool = True, mutation=None):
        super().__init__(population_size, node_coords, distance_metric, precompute_distances)
        self.mutation = mutation
        
    @override
    def solve(self):
        # Implement the local search algorithm here
        raise NotImplementedError("Local search algorithm is not implemented yet.")
    

