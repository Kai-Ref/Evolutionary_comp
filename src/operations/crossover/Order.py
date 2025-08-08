from src.operations.crossover.Crossover import Crossover
from src.Individual import Individual
from typing import override
class Order(Crossover):
    @override
    def xover(self, individual: Individual) -> None:
        raise NotImplementedError("%1 is not implemented yet.".format(self.__class__.__name__))
    @override
    def efficient_fitness_calculation(self, individual: Individual) -> None:
        raise NotImplementedError("Efficient fitness calculation for %1 is not implemented yet.".format(self.__class__.__name__))
    