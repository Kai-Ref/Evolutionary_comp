from src.operations.mutation.Mutation import Mutation
from src.Individual import Individual
from typing import override
class Inversion(Mutation):
    @override
    def mutate_individual(self, individual: Individual, i: int, j: int) -> None:
        raise NotImplementedError("inversion mutation is not implemented yet.")
    @override
    def efficient_fitness_calculation(self, individual: Individual, i: int, j: int) -> None:
        raise NotImplementedError("Efficient fitness calculation for inversion mutation is not implemented yet.")
