from src.operations.mutation.Mutation import Mutation
from src.Individual import Individual
from typing import override
import numpy as np

class Exchange(Mutation):
    @override
    def mutate_individual(self, individual: Individual, i: int, j: int, update_fitness:bool = True) -> Individual:
        """
        Performs Exchange mutation by swapping cities at positions i and j.

        Args:
            individual (Individual): Current individual.
            i (int): first index
            j (int): second index
            update_fitness (bool): whether to calculate the fitness for the new permutation or not. 
                This is mainly used to save computation time, when we already calculated the fitness beforehand.

        Returns:
            Individual: the newly permuted individual
        """
        n = len(individual.permutation)
        assert 0 <= i < n, "Index i is out of bounds."
        assert 0 <= j < n, "Index j is out of bounds."
        assert i != j, "Indices i and j must be different."
        
        # Swap cities at positions i and j
        new_tour = individual.permutation.copy()  # Copy the originaltour
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]

        # Create a new individual and compute its fitness efficiently
        new_individual = Individual(permutation=new_tour, tsp=individual.tsp)
        new_individual.fitness = individual.fitness + self.efficient_fitness_calculation(individual, i, j) if update_fitness else None
        return new_individual
    
    @override
    def efficient_fitness_calculation(self, individual: Individual, i: int, j: int) -> float:
        """
        Calculation of the deltas, i.e. the additive update to the fitness value 
        based solely on the modified edges, for Exchange mutation.

        This is based on this idea:
        For the symmetric TSP, Exchange only modifies those edges, which are connected to the two swapped nodes.
        Therefore we only need to (re-)calculate the distances for edges connected to those nodes.

        Args:
            individual (Individual): Current individual.
            i (int): first indice
            j (int): second indice

        Returns:
            float: delta value
        """
        tsp = individual.tsp
        tour = individual.permutation
        n = len(tour)
        if i == j:
            return 0.0
        if i > j:
            i, j = j, i
        
        # 1. Select the nodes involved in the exchange
        # We use  %n to wrap around the tour since it is circular (last city connects back to first)
        # So when i is 0, we correctly select n-1 (the last city in the tour), as its previously connected node
        city_i = tour[i]
        i_previous = tour[(i - 1) % n]
        i_next = tour[(i + 1) % n]

        city_j = tour[j]
        j_previous = tour[(j - 1) % n]
        j_next = tour[(j + 1) % n]

        # 2. Calculate the distances of modified edges, while considering 3 special cases.
        if j == i + 1: # Nodes that are adjacent (with the exception of i=0 and j=n-1) -> 3 edges modified
            old_distance = tsp.distance(i_previous, city_i) + tsp.distance(city_i, city_j) + tsp.distance(city_j, j_next)
            new_distance = tsp.distance(i_previous, city_j) + tsp.distance(city_j, city_i) + tsp.distance(city_i, j_next)
        elif i == 0 and j == n - 1: # first to last adjacency -> 3 (different) edges modified
            old_distance = tsp.distance(j_previous, city_j) + tsp.distance(city_j, city_i) + tsp.distance(city_i, i_next)
            new_distance = tsp.distance(j_previous, city_i) + tsp.distance(city_i, city_j) + tsp.distance(city_j, i_next)
        else: # Non adjacent nodes -> 4 edges modified
            old_distance = tsp.distance(i_previous, city_i) + tsp.distance(city_i, i_next) \
                        + tsp.distance(j_previous, city_j) + tsp.distance(city_j, j_next)
            new_distance = tsp.distance(i_previous, city_j) + tsp.distance(city_j, i_next) \
                        + tsp.distance(j_previous, city_i) + tsp.distance(city_i, j_next)

        # 3. Compute the difference
        return new_distance - old_distance


    def efficient_fitness_calculation_vectorized(self, individual: Individual, indices: np.ndarray) -> np.ndarray:
        """
        Vectorized calculation of the deltas, i.e. the additive update to the fitness value 
        based solely on the modified edges, for Exchange mutation.
        
        This is based on this idea:
        For the symmetric TSP, Exchange only modifies those edges, which are connected to the two swapped nodes.
        Therefore we only need to (re-)calculate the distances for edges connected to those nodes.

        Args:
            individual (Individual): Current individual.
            indices (np.ndarray): 2D-numpy array of the indices we want to modify.

        Returns:
            np.ndarray: array of all delta values, in the same order as the corresponding indices.
        """
        tour = np.array(individual.permutation, dtype=int)
        dist = individual.tsp.get_distance_matrix()
        n = len(tour)

        indices = np.array(list(indices), dtype=int)
        i_arr = indices[:, 0].copy()
        j_arr = indices[:, 1].copy()

        # Ensure i < j
        swap_mask = i_arr > j_arr
        i_arr[swap_mask], j_arr[swap_mask] = j_arr[swap_mask], i_arr[swap_mask]

        # 1. Select the nodes involved in the exchange
        # We use  %n to wrap around the tour since it is circular (last city connects back to first)
        # So when i is 0, we correctly select n-1 (the last city in the tour), as its previously connected node
        city_i = tour[i_arr]
        city_j = tour[j_arr]
        i_prev = tour[(i_arr - 1) % n]
        i_next = tour[(i_arr + 1) % n]
        j_prev = tour[(j_arr - 1) % n]
        j_next = tour[(j_arr + 1) % n]

        deltas = np.empty(len(i_arr), dtype=float)

        # 2. Calculate the distances of modified edges, while considering 3 special cases.
        # Since we are working with vectors, we use masks
        mask_adj = (j_arr == i_arr + 1)            # adjacent (with the exception of i=0 and j=n-1)
        mask_ij = (i_arr == 0) & (j_arr == n - 1)  # first-last adjacency
        mask_nadj = ~(mask_adj | mask_ij) # non adjacent

        # Nodes that are adjacent (with the exception of i=0 and j=n-1) -> 3 edges modified
        if np.any(mask_adj):
            old_adj = dist[i_prev[mask_adj], city_i[mask_adj]] + \
                    dist[city_i[mask_adj], city_j[mask_adj]] + \
                    dist[city_j[mask_adj], j_next[mask_adj]]
            new_adj = dist[i_prev[mask_adj], city_j[mask_adj]] + \
                    dist[city_j[mask_adj], city_i[mask_adj]] + \
                    dist[city_i[mask_adj], j_next[mask_adj]]
            deltas[mask_adj] = new_adj - old_adj

        # first to last adjacency -> 3 (different) edges modified
        if np.any(mask_ij):
            old_fl = dist[j_prev[mask_ij], city_j[mask_ij]] + \
                    dist[city_j[mask_ij], city_i[mask_ij]] + \
                    dist[city_i[mask_ij], i_next[mask_ij]]
            new_fl = dist[j_prev[mask_ij], city_i[mask_ij]] + \
                    dist[city_i[mask_ij], city_j[mask_ij]] + \
                    dist[city_j[mask_ij], i_next[mask_ij]]
            deltas[mask_ij] = new_fl - old_fl

        # Non-adjacent nodes -> 4 edges modified
        if np.any(mask_nadj):
            old_non = dist[i_prev[mask_nadj], city_i[mask_nadj]] + \
                    dist[city_i[mask_nadj], i_next[mask_nadj]] + \
                    dist[j_prev[mask_nadj], city_j[mask_nadj]] + \
                    dist[city_j[mask_nadj], j_next[mask_nadj]]
            new_non = dist[i_prev[mask_nadj], city_j[mask_nadj]] + \
                    dist[city_j[mask_nadj], i_next[mask_nadj]] + \
                    dist[j_prev[mask_nadj], city_i[mask_nadj]] + \
                    dist[city_i[mask_nadj], j_next[mask_nadj]]
            deltas[mask_nadj] = new_non - old_non
        return deltas

    @override
    def __str__(self):
        return "Exchange"
