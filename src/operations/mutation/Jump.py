from src.operations.mutation.Mutation import Mutation
from src.Individual import Individual
from typing import override
import numpy as np

class Jump(Mutation):
    @override
    def mutate_individual(self, individual: Individual, i: int, j: int) -> Individual:
        n = len(individual.permutation)
        assert 0 <= i < n, "Index i is out of bounds."
        assert 0 <= j < n, "Index j is out of bounds."
        assert i != j, "Indices i and j must be different."

        # Create a new tour by removing city at i and inserting it at j
        new_tour = individual.permutation.copy() # Copy the original tour
        city = new_tour.pop(i)
        new_tour.insert(j, city)

        # Removed, since it is cleaner to not overwrite the initial permutation
        # Update individual's permutation with the mutated tour
        # individual.permutation = new_tour
        return Individual(permutation=new_tour, tsp=individual.tsp)
    
    @override
    def efficient_fitness_calculation(self, individual: Individual, i: int, j: int) -> None:
        # detect the edges that were removed/added and compute their difference
        raise NotImplementedError("Efficient fitness calculation for jump mutation is not implemented yet.")
    