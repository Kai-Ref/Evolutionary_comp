from src.operations.mutation.Mutation import Mutation
from src.Individual import Individual
from typing import override
import numpy as np

class Jump(Mutation):
    @override
    def mutate_individual(self, individual: Individual, i: int, j: int, update_fitness: bool = False) -> Individual:
        n = len(individual.permutation)
        assert 0 <= i < n, "Index i is out of bounds."
        assert 0 <= j < n, "Index j is out of bounds."
        assert i != j, "Indices i and j must be different."
        
        # place i in position j and shift rest
        new_tour = individual.permutation.copy()  # Copy the original tour
        moved_city = new_tour.pop(i)
        new_tour.insert(j, moved_city)

        # Create a new individual and compute its fitness efficiently
        new_individual = Individual(permutation=new_tour, tsp=individual.tsp)
        # if update_fitness as this might not be required for EA
        new_individual.fitness = individual.fitness + self.efficient_fitness_calculation(individual, i, j) if update_fitness else None
        return new_individual
    
    @override
    def efficient_fitness_calculation(self, individual: Individual, i: int, j: int) -> float:
        tsp = individual.tsp
        tour = individual.permutation
        n = len(tour)

        city_i = tour[i]
        i_previous = tour[(i - 1) % n]
        i_next = tour[(i + 1) % n]

        city_j = tour[j]
        j_previous = tour[(j - 1) % n]
        j_next = tour[(j + 1) % n]


        if i > j:
            old_distance = tsp.distance(i_previous, city_i) + \
                        tsp.distance(city_i, i_next) + \
                        tsp.distance(j_previous, city_j)
                        
            new_distance = tsp.distance(i_previous, i_next) + \
                        tsp.distance(j_previous, city_i) + \
                        tsp.distance(city_i, city_j)
        else:
            old_distance = tsp.distance(i_previous, city_i) + \
                        tsp.distance(city_i, i_next) + \
                        tsp.distance(city_j, j_next)
                        
            new_distance = tsp.distance(i_previous, i_next) + \
                        tsp.distance(j_next, city_i) + \
                        tsp.distance(city_j, city_i)

        return new_distance - old_distance
    
    def efficient_fitness_calculation_vectorized(self, individual: Individual, indices: np.ndarray) -> np.ndarray:
        """Vectorized delta calculation for Jump mutation."""
        tour = np.array(individual.permutation, dtype=int)
        dist = individual.tsp.get_distance_matrix()
        n = len(tour)

        indices = np.array(list(indices), dtype=int)
        i_arr = indices[:, 0]
        j_arr = indices[:, 1]

        city_i = tour[i_arr]
        i_prev = tour[(i_arr - 1) % n]
        i_next = tour[(i_arr + 1) % n]

        city_j = tour[j_arr]
        j_prev = tour[(j_arr - 1) % n]
        j_next = tour[(j_arr + 1) % n]

        deltas = np.empty(len(i_arr), dtype=float)

        # Mask for i > j
        mask_ig = i_arr > j_arr

        # Case: i > j
        old_d_ig = dist[i_prev, city_i] + dist[city_i, i_next] + dist[j_prev, city_j]
        new_d_ig = dist[i_prev, i_next] + dist[j_prev, city_i] + dist[city_i, city_j]
        deltas[mask_ig] = new_d_ig[mask_ig] - old_d_ig[mask_ig]

        # Case: i <= j
        old_d_le = dist[i_prev, city_i] + dist[city_i, i_next] + dist[city_j, j_next]
        new_d_le = dist[i_prev, i_next] + dist[j_next, city_i] + dist[city_j, city_i]
        deltas[~mask_ig] = new_d_le[~mask_ig] - old_d_le[~mask_ig]

        return deltas

    @override
    def __str__(self):
        return "Jump"
