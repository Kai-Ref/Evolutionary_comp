from src.operations.mutation.Mutation import Mutation
from src.Individual import Individual
from typing import override
import numpy as np

class Jump(Mutation):
    @override
    def mutate_individual(self, individual: Individual, i: int, j: int, update_fitness: bool = False) -> Individual:
        """
        Performs Jump mutation from position i to j.

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
        
        # place i in position j and shift rest
        new_tour = individual.permutation.copy()  # Copy the original tour
        moved_city = new_tour.pop(i)
        new_tour.insert(j, moved_city)

        # Create a new individual and compute its fitness efficiently
        new_individual = Individual(permutation=new_tour, tsp=individual.tsp)
        new_individual.fitness = individual.fitness + self.efficient_fitness_calculation(individual, i, j) if update_fitness else None
        return new_individual
    
    @override
    def efficient_fitness_calculation(self, individual: Individual, i: int, j: int) -> float:
        """
        Calculation of the deltas, i.e. the additive update to the fitness value 
        based solely on the modified edges, for Jump mutation.

        Idea:
        In a Jump mutation, one city is removed from its position i and inserted at position j.
        This affects only the edges immediately before and after i, and the edges around j.
        By computing the difference between old and new distances for these affected edges,
        we can update the fitness efficiently without recomputing the whole tour.

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
        
        # special case: when i=0 and j=n-1, the tour distance remains unchanged (the same goes for the other way around)
        if (i==0 and j==(n - 1)) or (j==0 and i==(n - 1)):
            return 0.0
        
        # 1. Select the nodes involved in the jump
        # We use  %n to wrap around the tour since it is circular (last city connects back to first)
        # So when i is 0, we correctly select n-1 (the last city in the tour), as its previously connected node
        city_i = tour[i]
        i_previous = tour[(i - 1) % n]
        i_next = tour[(i + 1) % n]

        city_j = tour[j]
        j_previous = tour[(j - 1) % n]
        j_next = tour[(j + 1) % n]

        # 2. Calculate the distances of modified edges, while considering 2 special cases.
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
        # 3. Compute the difference
        return new_distance - old_distance
    
    def efficient_fitness_calculation_vectorized(self, individual: Individual, indices: np.ndarray) -> np.ndarray:
        """
        Vectorized calculation of the deltas, i.e. the additive update to the fitness value 
        based solely on the modified edges, for Jump mutation.

        
        Idea:
        In a Jump mutation, one city is removed from its position i and inserted at position j.
        This affects only the edges immediately before and after i, and the edges around j.
        By computing the difference between old and new distances for these affected edges,
        we can update the fitness efficiently without recomputing the whole tour.

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
        i_arr = indices[:, 0]
        j_arr = indices[:, 1]

        # 1. Select the nodes involved in the jump
        # We use  %n to wrap around the tour since it is circular (last city connects back to first)
        # So when i is 0, we correctly select n-1 (the last city in the tour), as its previously connected node
        city_i = tour[i_arr]
        i_prev = tour[(i_arr - 1) % n]
        i_next = tour[(i_arr + 1) % n]
        city_j = tour[j_arr]
        j_prev = tour[(j_arr - 1) % n]
        j_next = tour[(j_arr + 1) % n]

        deltas = np.empty(len(i_arr), dtype=float)

        # special case: when i=0 and j=n-1, the tour distance remains unchanged (the same goes for the other way around)
        adj_mask = ((i_arr == 0) & (j_arr == (n - 1))) | ((j_arr == 0) & (i_arr == (n - 1)))
        deltas[adj_mask] = 0.0

        # 2. Calculate the distances of modified edges, while considering 3 special cases.
        # Since we are working with vectors, we use masks
        mask_ij = (i_arr > j_arr) & (~adj_mask) # i >j
        mask_else = (~mask_ij) & (~adj_mask) # else

        if np.any(mask_ij):# Case: i > j
            old_d_ig = dist[i_prev, city_i] + dist[city_i, i_next] + dist[j_prev, city_j]
            new_d_ig = dist[i_prev, i_next] + dist[j_prev, city_i] + dist[city_i, city_j]
            deltas[mask_ij] = new_d_ig[mask_ij] - old_d_ig[mask_ij]

        # Case: i <= j
        if np.any(mask_else):
            old_d_le = dist[i_prev, city_i] + dist[city_i, i_next] + dist[city_j, j_next]
            new_d_le = dist[i_prev, i_next] + dist[city_j, city_i] + dist[city_i, j_next]
            deltas[mask_else] = new_d_le[mask_else] - old_d_le[mask_else]

        return deltas

    @override
    def __str__(self):
        return "Jump"
