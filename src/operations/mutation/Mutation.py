from src.Population import Population
from src.Individual import Individual
class Mutation:
    def __call__(self, population: Population, i: int, j: int) -> Population:
        self.mutation_probability = None #TODO: often either 1/population size or 1/length of permutation array
        for individual in population:
            print(f"Mutating individual {individual} at indices {i} and {j}.")
            self.mutate_individual(individual, i, j)
            print(f"Individual after mutation: {individual}")     
        return population
    
    def mutate_individual(self, individual: Individual, i: int, j: int, update_fitness: bool = False) -> None:
        raise NotImplementedError("Mutate method must be implemented in subclasses.")

    def efficient_fitness_calculation(self, individual: Individual, i: int, j: int) -> None:
        if individual.fitness is None:
            individual.calculate_fitness()

    def __str__(self):
        return NotImplementedError("__str__ method must be implemented in subclasses.")

    def __repr__(self):
        return self.__str__()
            