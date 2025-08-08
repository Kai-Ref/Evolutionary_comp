from src.Population import Population
from src.Individual import Individual
class Crossover:
    def __call__(self, population: Population) -> Population:
        for individual in population.individuals:
            self.xover(individual)
        return population

    def xover(self, individual: Individual) -> None:
        raise NotImplementedError("Crossover method must be implemented in subclasses.")

    def efficient_fitness_calculation(self, population: Population) -> None:
        raise NotImplementedError("Efficient fitness calculation must be implemented in subclasses.")