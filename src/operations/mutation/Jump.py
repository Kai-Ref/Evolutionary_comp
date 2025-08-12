from src.operations.mutation.Mutation import Mutation
from src.Individual import Individual
from typing import override
class Jump(Mutation):
    @override
    def mutate(self, individual: Individual, i: int, j: int) -> None:
        raise NotImplementedError("Jump mutation is not implemented yet.")
    @override
    def efficient_fitness_calculation(self, individual: Individual, i: int, j: int) -> None:
        raise NotImplementedError("Efficient fitness calculation for jump mutation is not implemented yet.")
    