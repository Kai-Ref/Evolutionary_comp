from src.operations.mutation.Mutation import Mutation
from src.Individual import Individual
from typing import override
import numpy as np

class Jump(Mutation):
    @override
    def mutate(self, individual: Individual, i: int, j: int) -> None:
        n = len(individual.permutation)
        assert 0 <= i < n, "Index i is out of bounds."
        assert 0 <= j < n, "Index j is out of bounds."
        assert i != j, "Indices i and j must be different."

        # Create a new tour by removing city at i and inserting it at j
        new_tour = individual.permutation.tolist()  # Copy the original tour
        city = new_tour.pop(i)
        new_tour.insert(j, city)

        # Update individual's permutation with the mutated tour
        individual.permutation = np.array(new_tour)
    
    @override
    def efficient_fitness_calculation(self, individual: Individual, i: int, j: int) -> None:
        raise NotImplementedError("Efficient fitness calculation for jump mutation is not implemented yet.")
    