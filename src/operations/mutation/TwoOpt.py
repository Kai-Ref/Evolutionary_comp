from src.operations.mutation.Mutation import Mutation
from src.Individual import Individual
from typing import override
class TwoOpt(Mutation):
    @override
    def mutate_individual(self, individual: Individual, i: int, j: int, update_fitness: bool = True) -> None:
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
        tsp = individual.tsp
        tour = individual.permutation
        n = len(tour)

        # Handle i > j for proper segment reversal
        if i > j:
            i, j = j, i

        # When considering the full reversal (i=0 and j=n-1) in the symmetric TSP, the total distance remains unchanged.
        # So we just return the a delta of 0.
        if i == 0 and j == n - 1:
            return 0.0


        # For the symmetric TSP, 2-Opt only modifies two edges, which are connected to the rest of the tour.
        # Specifically, the edges at the very beginning and end of the segment.
        # Therefore we only need to (re-)calculate the distances for four edges in total.

        # 1. Select the changed nodes and their connecting node (i-1)/(j+1)
        # We use  %n to wrap around the tour since it is circular (last city connects back to first)
        # So when i is 0, we correctly select n-1 (the last city in the tour), as its previously connected node
        node_before_i = tour[(i - 1) % n]  
        node_i = tour[i]            
        node_j = tour[j]            
        node_after_j = tour[(j + 1) % n] 

        # 2. Calculate the old and new distances of the two modified edges
        old_distance = tsp.distance(node_before_i, node_i) + tsp.distance(node_j, node_after_j)
        new_distance = tsp.distance(node_before_i, node_j) + tsp.distance(node_i, node_after_j)

        # 3. Compute the difference and add it to the previous fitness
        return new_distance - old_distance



    # @override
    # def efficient_fitness_calculation(self, old_individual: Individual, new_individual: Individual, i: int, j: int) -> float:
    #     n = len(old_individual.permutation)
    #     tsp = old_individual.tsp
        
    #     # Handle i > j for proper segment reversal
    #     if i > j:
    #         i, j = j, i

    #     # When considering the full reversal (i=0 and j=n-1) in the symmetric TSP, the total distance remains unchanged.
    #     # So we just return the original fitness.
    #     if i == 0 and j == n - 1:
    #         return old_individual.fitness

    #     old_permutation = old_individual.permutation.copy()


    #     # For the symmetric TSP, 2-Opt only modifies two edges, which are connected to the rest of the tour.
    #     # Specifically, the edges at the very beginning and end of the segment.
    #     # Therefore we only need to (re-)calculate the distances for four edges in total.

    #     # 1. Select the changed nodes and their connecting node (i-1)/(j+1)
    #     # We use  %n to wrap around the tour since it is circular (last city connects back to first)
    #     # So when i is 0, we correctly select n-1 (the last city in the tour), as its previously connected node
    #     node_before_i = old_permutation[(i - 1) % n]  
    #     node_i = old_permutation[i]            
    #     node_j = old_permutation[j]            
    #     node_after_j = old_permutation[(j + 1) % n] 

    #     # 2. Calculate the old and new distances of the two modified edges
    #     old_distance = tsp.distance(node_before_i, node_i) + tsp.distance(node_j, node_after_j)
    #     new_distance = tsp.distance(node_before_i, node_j) + tsp.distance(node_i, node_after_j)

    #     # 3. Compute the difference and add it to the previous fitness
    #     delta = new_distance - old_distance
    #     return old_individual.fitness + delta
