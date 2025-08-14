from src.operations.mutation.Mutation import Mutation
from src.Individual import Individual
from typing import override
import numpy as np

class Exchange(Mutation):
    @override
    def mutate_individual(self, individual: Individual, i: int, j: int, update_fitness:bool = True) -> None:
        n = len(individual.permutation)
        assert 0 <= i < n, "Index i is out of bounds."
        assert 0 <= j < n, "Index j is out of bounds."
        assert i != j, "Indices i and j must be different."
        
        # Swap cities at positions i and j
        new_tour = individual.permutation.copy()  # Copy the originaltour
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]

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
        if i == j:
            return 0.0
        if i > j:
            i, j = j, i
        
        # For the symmetric TSP, Exchange only modifies those edges, which are connected to the two swapped nodes.
        # Therefore we only need to (re-)calculate the distances for edges connected to those nodes.

        # 1. Select the nodes involved in the exchange
        # We use  %n to wrap around the tour since it is circular (last city connects back to first)
        # So when i is 0, we correctly select n-1 (the last city in the tour), as its previously connected node
        city_i = tour[i]
        i_previous = tour[(i - 1) % n]
        i_next = tour[(i + 1) % n]

        city_j = tour[j]
        j_previous = tour[(j - 1) % n]
        j_next = tour[(j + 1) % n]

        # 2. Calculate the distances of modified edges
        # if (j == (i + 1) % n) or (i == 0 and j == n - 1):
        #     old_distance = tsp.distance(i_previous, city_i) + tsp.distance(city_i, city_j) \
        #                 + tsp.distance(city_j, j_next)
        #     new_distance = tsp.distance(i_previous, city_j) + tsp.distance(city_j, city_i) \
        #                 + tsp.distance(city_i, j_next)
        if j == i + 1:
            # normal adjacent
            old_distance = tsp.distance(i_previous, city_i) + tsp.distance(city_i, city_j) + tsp.distance(city_j, j_next)
            new_distance = tsp.distance(i_previous, city_j) + tsp.distance(city_j, city_i) + tsp.distance(city_i, j_next)
        elif i == 0 and j == n - 1:
            # first-last adjacent
            old_distance = tsp.distance(j_previous, city_j) + tsp.distance(city_j, city_i) + tsp.distance(city_i, i_next)
            new_distance = tsp.distance(j_previous, city_i) + tsp.distance(city_i, city_j) + tsp.distance(city_j, i_next)
        else:
            old_distance = tsp.distance(i_previous, city_i) + tsp.distance(city_i, i_next) \
                        + tsp.distance(j_previous, city_j) + tsp.distance(city_j, j_next)
            new_distance = tsp.distance(i_previous, city_j) + tsp.distance(city_j, i_next) \
                        + tsp.distance(j_previous, city_i) + tsp.distance(city_i, j_next)

        # 3. Compute the difference and add it to the previous fitness
        return new_distance - old_distance


    def efficient_fitness_calculation_vectorized(self, individual: Individual, indices: np.ndarray) -> np.ndarray:
        tour = np.array(individual.permutation, dtype=int)
        dist = individual.tsp.get_distance_matrix()
        n = len(tour)

        indices = np.array(list(indices), dtype=int)
        i_arr = indices[:, 0].copy()
        j_arr = indices[:, 1].copy()

        # Ensure i < j
        swap_mask = i_arr > j_arr
        i_arr[swap_mask], j_arr[swap_mask] = j_arr[swap_mask], i_arr[swap_mask]

        # Get city numbers
        city_i = tour[i_arr]
        city_j = tour[j_arr]

        # Circular neighbors
        i_prev = tour[(i_arr - 1) % n]
        i_next = tour[(i_arr + 1) % n]
        j_prev = tour[(j_arr - 1) % n]
        j_next = tour[(j_arr + 1) % n]

        deltas = np.empty(len(i_arr), dtype=float)
        # Masks
        adj_mask = (j_arr == i_arr + 1)            # normal adjacent
        first_last_mask = (i_arr == 0) & (j_arr == n - 1)  # first-last adjacency
        non_adj_mask = ~(adj_mask | first_last_mask)

        # 1. Normal adjacent swaps
        if np.any(adj_mask):
            old_adj = dist[i_prev[adj_mask], city_i[adj_mask]] + \
                    dist[city_i[adj_mask], city_j[adj_mask]] + \
                    dist[city_j[adj_mask], j_next[adj_mask]]
            new_adj = dist[i_prev[adj_mask], city_j[adj_mask]] + \
                    dist[city_j[adj_mask], city_i[adj_mask]] + \
                    dist[city_i[adj_mask], j_next[adj_mask]]
            deltas[adj_mask] = new_adj - old_adj

        # 2. First-last swaps
        if np.any(first_last_mask):
            old_fl = dist[j_prev[first_last_mask], city_j[first_last_mask]] + \
                    dist[city_j[first_last_mask], city_i[first_last_mask]] + \
                    dist[city_i[first_last_mask], i_next[first_last_mask]]
            new_fl = dist[j_prev[first_last_mask], city_i[first_last_mask]] + \
                    dist[city_i[first_last_mask], city_j[first_last_mask]] + \
                    dist[city_j[first_last_mask], i_next[first_last_mask]]
            deltas[first_last_mask] = new_fl - old_fl

        # 3. Non-adjacent swaps
        if np.any(non_adj_mask):
            old_non = dist[i_prev[non_adj_mask], city_i[non_adj_mask]] + \
                    dist[city_i[non_adj_mask], i_next[non_adj_mask]] + \
                    dist[j_prev[non_adj_mask], city_j[non_adj_mask]] + \
                    dist[city_j[non_adj_mask], j_next[non_adj_mask]]
            new_non = dist[i_prev[non_adj_mask], city_j[non_adj_mask]] + \
                    dist[city_j[non_adj_mask], i_next[non_adj_mask]] + \
                    dist[j_prev[non_adj_mask], city_i[non_adj_mask]] + \
                    dist[city_i[non_adj_mask], j_next[non_adj_mask]]
            deltas[non_adj_mask] = new_non - old_non
        return deltas

    @override
    def __str__(self):
        return "Exchange"


    
    # @override
    # def efficient_fitness_calculation(self, old_individual: Individual, new_individual: Individual, i: int, j: int) -> float:
    #     # Can be made more memory-efficient, probably not needed though.
    #     n = len(old_individual.permutation)
    #     old_permutation = old_individual.permutation.copy()
    #     new_permutation = new_individual.permutation.copy()
    #     tsp = old_individual.tsp

    #     # For the symmetric TSP, Exchange only modifies those edges, which are connected to the two swapped nodes.
    #     # Therefore we only need to (re-)calculate the distances for edges connected to those nodes.

    #     # 1. Select the old nodes and calculate their distances
    #     # We use  %n to wrap around the tour since it is circular (last city connects back to first)
    #     # So when i is 0, we correctly select n-1 (the last city in the tour), as its previously connected node
    #     old_city_i = old_permutation[i]
    #     old_i_previous = old_permutation[(i - 1) % n]
    #     old_i_next = old_permutation[(i + 1) % n]

    #     old_city_j = old_permutation[j]
    #     old_j_previous = old_permutation[(j - 1) % n]
    #     old_j_next = old_permutation[(j + 1) % n]

    #     old_distance = tsp.distance(old_i_previous, old_city_i) + tsp.distance(old_city_i, old_i_next) \
    #                  + tsp.distance(old_j_previous, old_city_j) + tsp.distance(old_city_j, old_j_next)

    #     # 2. Select the new nodes and calculate their distances
    #     new_city_i = new_permutation[i]
    #     new_i_previous = new_permutation[(i - 1) % n]
    #     new_i_next = new_permutation[(i + 1) % n]

    #     new_city_j = new_permutation[j]
    #     new_j_previous = new_permutation[(j - 1) % n]
    #     new_j_next = new_permutation[(j + 1) % n]

    #     # Sum of new edges
    #     new_distance = tsp.distance(new_i_previous, new_city_i) + tsp.distance(new_city_i, new_i_next) \
    #                  + tsp.distance(new_j_previous, new_city_j) + tsp.distance(new_city_j, new_j_next)

    #      # 3. Compute the difference and add it to the previous fitness
    #     delta = new_distance - old_distance
    #     return old_individual.fitness + delta
    


