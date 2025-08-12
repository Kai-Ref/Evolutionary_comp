import numpy as np
from typing import override
from src.TSP import TSP
from src.Population import Population
class LocalSearch(TSP):
    def __init__(self, filepath: str, distance_metric: str = 'euclidean', precompute_distances: bool = True, mutation=None, population_size: int = 1):
        super().__init__(filepath=filepath, distance_metric=distance_metric, precompute_distances=precompute_distances, population_size=population_size, mutation=mutation)
        
    @override
    def solve(self, max_iterations: int = 1E4):
        pass
        
    
    def get_next_neighbour(self, current: Population) -> Population:
        """
        Generator Function which returns the next neighbour. It returns None if all possible neighbours where already returned
        """
        neighbour = None

    def jump_neighbors(tour: list) -> list:
        """
        Perform the Jump operation between two cities in the tour.
        """
        n = len(tour)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Create a new tour by removing city at i and inserting it at j
                new_tour = tour[:]
                city = new_tour.pop(i)
                new_tour.insert(j, city)
                yield new_tour

    
    def exchange_neighbors(tour: list) -> list:
        """
        Generate neighbors by swapping two cities in the tour.
        """
        n = len(tour)
        for i in range(n):
            for j in range(i + 1, n):
                new_tour = tour[:]
                # Swap cities at positions i and j
                new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
                yield new_tour


    def Two_Opt(tour: list) -> list:
        """
        Generate neighbors by reversing the segment between two indices (2-opt move).
        """
        n = len(tour)
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_tour = tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]
                yield new_tour

    
