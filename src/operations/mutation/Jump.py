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

        # if abs(i-j)==1:
        #     # print(i, j)
        #     return 0.0
        
        # special case: when i=0 and j=n-1, the tour distance remains unchanged (the same goes for the other way around)
        if (i==0 and j==(n - 1)) or (j==0 and i==(n - 1)):
            return 0.0
        
        
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
                        tsp.distance(city_j, city_i) + \
                        tsp.distance(city_i, j_next) 

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

        # special-case wrap-around adjacency (i=0, j=n-1) or (j=0, i=n-1)
        adj_mask = ((i_arr == 0) & (j_arr == (n - 1))) | ((j_arr == 0) & (i_arr == (n - 1)))
        deltas[adj_mask] = 0.0

        # Mask for i > j  (parentheses are necessary!)
        mask_ig = (i_arr > j_arr) & (~adj_mask)
        mask_le = (~mask_ig) & (~adj_mask)

        # Case: i > j
        old_d_ig = dist[i_prev, city_i] + dist[city_i, i_next] + dist[j_prev, city_j]
        new_d_ig = dist[i_prev, i_next] + dist[j_prev, city_i] + dist[city_i, city_j]
        deltas[mask_ig] = new_d_ig[mask_ig] - old_d_ig[mask_ig]

        # Case: i <= j
        old_d_le = dist[i_prev, city_i] + dist[city_i, i_next] + dist[city_j, j_next]
        # <-- fixed ordering here: city_i then j_next (matches scalar version)
        new_d_le = dist[i_prev, i_next] + dist[city_j, city_i] + dist[city_i, j_next]

        deltas[mask_le] = new_d_le[mask_le] - old_d_le[mask_le]

        return deltas

    @override
    def __str__(self):
        return "Jump"
