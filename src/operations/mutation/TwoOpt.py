from src.operations.mutation.Mutation import Mutation
from src.Individual import Individual
from typing import override
import numpy as np

class TwoOpt(Mutation):
    @override
    def mutate_individual(self, individual: Individual, i: int, j: int, update_fitness: bool = True) -> Individual:
        """
        Performs 2-Opt mutation from node i to j.

        Args:
            individual (Individual): Current individual.
            i (int): first indice
            j (int): second indice
            update_fitness (bool): whether to calculate the fitness for the new permutation or not. 
                This is mainly used to save computation time, when we already calculated the fitness beforehand.

        Returns:
            Individual: the newly permuted individual
        """
        n = len(individual.permutation)
        assert 0 <= i < n, "Index i is out of bounds."
        assert 0 <= j < n, "Index j is out of bounds."
        assert i != j, "Indices i and j must be different."
        
        # Handle i > j for proper segment reversal
        if i > j:
            i, j = j, i
        
        # Reverse the segment between i and j (inclusive)
        new_tour = individual.permutation.copy()
        new_tour[i:j+1] = reversed(new_tour[i:j+1])

        # Create a new individual and compute its fitness efficiently
        new_individual = Individual(permutation=new_tour, tsp=individual.tsp)
        new_individual.fitness = individual.fitness + self.efficient_fitness_calculation(individual, i, j) if update_fitness else None
        return new_individual


    @override
    def efficient_fitness_calculation(self, individual: Individual, i: int, j: int) -> float:
        """
        Calculation of the deltas, i.e. the additive update to the fitness value 
        based solely on the modified edges, for 2-Opt mutation.

        This is based on this idea:
        For the symmetric TSP, 2-Opt only modifies two edges, which are connected to the rest of the tour.
        Specifically, the edges at the very beginning and end of the segment.
        Therefore we only need to (re-)calculate the distances for four edges in total. 
        However, some specific cases also need to be handled.

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

        # Handle i > j for proper segment reversal
        if i > j:
            i, j = j, i

        # 1. Select the changed nodes and their connecting node (i-1)/(j+1)
        # We use  %n to wrap around the tour since it is circular (last city connects back to first)
        # So when i is 0, we correctly select n-1 (the last city in the tour), as its previously connected node
        node_before_i = tour[(i - 1) % n]  
        node_i = tour[i]            
        node_j = tour[j]            
        node_after_j = tour[(j + 1) % n] 

        # 2. When considering the full reversal (i=0 and j=n-1) in the symmetric TSP, the total distance remains unchanged.
        # So we just return the a delta of 0.
        if i == 0 and j == n - 1:
            return 0.0

        # 3. Calculate the old and new distances of the two modified edges
        old_distance = tsp.distance(node_before_i, node_i) + tsp.distance(node_j, node_after_j)
        new_distance = tsp.distance(node_before_i, node_j) + tsp.distance(node_i, node_after_j)

        # 4. Compute the difference
        return new_distance - old_distance

    def efficient_fitness_calculation_vectorized(self, individual: Individual, indices: np.ndarray) -> np.ndarray:
        """
        Vectorized calculation of the deltas, i.e. the additive update to the fitness value 
        based solely on the modified edges, for 2-Opt mutation.

        This is based on this idea:
        For the symmetric TSP, 2-Opt only modifies two edges, which are connected to the rest of the tour.
        Specifically, the edges at the very beginning and end of the segment.
        Therefore we only need to (re-)calculate the distances for four edges in total. 
        However, some specific cases also need to be handled. 

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

        # Ensure i < j for segment reversal
        swap_mask = i_arr > j_arr
        i_arr[swap_mask], j_arr[swap_mask] = j_arr[swap_mask], i_arr[swap_mask]

        # 1. Select the changed nodes and their connecting node (i-1)/(j+1)
        # We use  %n to wrap around the tour since it is circular (last city connects back to first)
        # So when i is 0, we correctly select n-1 (the last city in the tour), as its previously connected node
        node_before_i = tour[(i_arr - 1) % n]
        node_i = tour[i_arr]
        node_j = tour[j_arr]
        node_after_j = tour[(j_arr + 1) % n]

        deltas = np.empty(len(i_arr), dtype=float)

        # 2. When considering the full reversal (i=0 and j=n-1) in the symmetric TSP, the total distance remains unchanged.
        # So we just set delta to 0.
        full_rev_mask = (i_arr == 0) & (j_arr == n - 1)
        deltas[full_rev_mask] = 0.0

        # 3. Calculate the old and new distances of the two modified edges
        mask = ~full_rev_mask
        old_d = dist[node_before_i, node_i] + dist[node_j, node_after_j]
        new_d = dist[node_before_i, node_j] + dist[node_i, node_after_j]

        # 4. Compute the difference
        deltas[mask] = new_d[mask] - old_d[mask]

        return deltas

    
    @override
    def __str__(self):
        return "TwoOpt"