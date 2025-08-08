from src.Population import Population
from src.Individual import Individual
class Mutation:
    def __call__(self, population: Population) -> Population:
        for individual in population.individuals:
            self.mutate(individual)
            self.efficient_fitness_calculation(individual)
        return population
    
    def mutate(self, individual: Individual) -> None:
        raise NotImplementedError("Mutate method must be implemented in subclasses.")
        
    def efficient_fitness_calculation(self, individual: Individual) -> None:
        raise NotImplementedError("Efficient fitness calculation must be implemented in subclasses.")