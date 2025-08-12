from src.operations.mutation.Mutation import Mutation
from src.Individual import Individual
from typing import override
class Exchange(Mutation):
    @override
    def mutate(self, individual: Individual, i: int, j: int) -> None:
        n = len(individual.permutation)
        assert 0 <= i < n, "Index i is out of bounds."
        assert 0 <= j < n, "Index j is out of bounds."
        assert i != j, "Indices i and j must be different."
        
        # Swap cities at positions i and j
        new_tour = individual.permutation.tolist()  # Copy the original tour
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        individual.permutation = new_tour
        
    @override
    def efficient_fitness_calculation(self, individual: Individual, i: int, j: int) -> None:
        raise NotImplementedError("Efficient fitness calculation for exchange mutation is not implemented yet.")