from src.operations.mutation.Mutation import Mutation
from src.Individual import Individual
from typing import override
class Exchange(Mutation):
    @override
    def mutate_individual(self, individual: Individual, i: int, j: int) -> None:
        n = len(individual.permutation)
        assert 0 <= i < n, "Index i is out of bounds."
        assert 0 <= j < n, "Index j is out of bounds."
        assert i != j, "Indices i and j must be different."
        
        # Swap cities at positions i and j
        new_tour = individual.permutation.copy()  # Copy the originaltour
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]

        # Create a new individual and compute its fitness efficiently
        new_individual = Individual(permutation=new_tour, tsp=individual.tsp)
        new_individual.fitness = self.efficient_fitness_calculation(individual, new_individual, i, j)
        return new_individual
    
    @override
    def efficient_fitness_calculation(self, old_individual: Individual, new_individual: Individual, i: int, j: int) -> float:
        n = len(old_individual.permutation)
        old_permutation = old_individual.permutation.copy()
        new_permutation = new_individual.permutation.copy()
        tsp = old_individual.tsp

        # For the symmetric TSP, Exchange only modifies those edges, which are connected to the two swapped nodes.
        # Therefore we only need to (re-)calculate the distances for edges connected to those nodes.

        # 1. Select the old nodes and calculate their distances
        # We use  %n to wrap around the tour since it is circular (last city connects back to first)
        # So when i is 0, we correctly select n-1 (the last city in the tour), as its previously connected node
        old_city_i = old_permutation[i]
        old_i_previous = old_permutation[(i - 1) % n]
        old_i_next = old_permutation[(i + 1) % n]

        old_city_j = old_permutation[j]
        old_j_previous = old_permutation[(j - 1) % n]
        old_j_next = old_permutation[(j + 1) % n]

        old_distance = tsp.distance(old_i_previous, old_city_i) + tsp.distance(old_city_i, old_i_next) \
                     + tsp.distance(old_j_previous, old_city_j) + tsp.distance(old_city_j, old_j_next)

        # 2. Select the new nodes and calculate their distances
        new_city_i = new_permutation[i]
        new_i_previous = new_permutation[(i - 1) % n]
        new_i_next = new_permutation[(i + 1) % n]

        new_city_j = new_permutation[j]
        new_j_previous = new_permutation[(j - 1) % n]
        new_j_next = new_permutation[(j + 1) % n]

        # Sum of new edges
        new_distance = tsp.distance(new_i_previous, new_city_i) + tsp.distance(new_city_i, new_i_next) \
                     + tsp.distance(new_j_previous, new_city_j) + tsp.distance(new_city_j, new_j_next)

         # 3. Compute the difference and add it to the previous fitness
        delta = new_distance - old_distance
        return old_individual.fitness + delta

