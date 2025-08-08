from src.operations.mutation.Mutation import Mutation
from src.Individual import Individual
from typing import override
class Exchange(Mutation):
    @override
    def mutate(self, individual: Individual) -> None:
        raise NotImplementedError("exchange mutation is not implemented yet.")
    @override
    def efficient_fitness_calculation(self, individual: Individual) -> None:
        raise NotImplementedError("Efficient fitness calculation for exchange mutation is not implemented yet.")