from src.operations.mutation.Mutation import Mutation
from src.Individual import Individual
from typing import override
class TwoOpt(Mutation):
    @override
    def mutate_individual(self, individual: Individual, i: int, j: int) -> None:
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

        # Update individual's permutation with the mutated tour
        # individual.permutation = new_tour
        return Individual(permutation=new_tour, tsp=individual.tsp)

    @override
    def efficient_fitness_calculation(self, individual: Individual, i: int, j: int) -> None:
        raise NotImplementedError("Efficient fitness calculation for Two Opt mutation is not implemented yet.")