from src.Population import Population
from src.Individual import Individual
class Mutation:
    def __call__(self, individual: Individual, i: int, j: int) -> Individual:
        self.mutation_probability = None #TODO: often either 1/population size or 1/length of permutation array
        # for idx in range(len(population)):
        #     individual = population[idx]
        print(f"Mutating individual {individual} at indices {i} and {j}.")
        self.mutate_individual(individual, i, j)
        print(f"Individual after mutation: {individual}")
        # self.efficient_fitness_calculation(individual, i, j)
        return individual
    
    def mutate_individual(self, individual: Individual, i: int, j: int) -> None:
        raise NotImplementedError("Mutate method must be implemented in subclasses.")

    def efficient_fitness_calculation(self, individual: Individual, i: int, j: int) -> None:
        if individual.fitness is None:
            individual.calculate_fitness()
            