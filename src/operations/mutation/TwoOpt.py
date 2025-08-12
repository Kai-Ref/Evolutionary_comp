from src.operations.mutation.Mutation import Mutation
from src.Individual import Individual
from typing import override
class TwoOpt(Mutation):
    @override
    def mutate(self, individual: Individual, i: int, j: int) -> None:
        raise NotImplementedError("Two Opt mutation is not implemented yet.")
    @override
    def efficient_fitness_calculation(self, individual: Individual, i: int, j: int) -> None:
        raise NotImplementedError("Efficient fitness calculation for Two Opt mutation is not implemented yet.")