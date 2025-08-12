from src.Population import Population
from src.Individual import Individual
class Mutation:
    def __call__(self, population: Population, i: int, j: int) -> Population:
        for individual in population.individuals:
            self.mutate(individual, i, j)
            self.efficient_fitness_calculation(individual)
        return population
    
    def mutate_individual(self, individual: Individual, i: int, j: int) -> None:
        raise NotImplementedError("Mutate method must be implemented in subclasses.")

    def efficient_fitness_calculation(self, individual: Individual, i: int, j: int) -> None:
        raise NotImplementedError("Efficient fitness calculation must be implemented in subclasses.")