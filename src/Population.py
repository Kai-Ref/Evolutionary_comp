import numpy as np
from src.Individual import Individual

class Population:
    def __init__(self, population_size: int, number_of_nodes: int):
        self.population_size = population_size
        self.individuals = [Individual(number_of_nodes) for _ in range(population_size)]
        
    def get_mean_fitness(self) -> float:
        total_fitness = sum(ind.fitness for ind in self.individuals if ind.fitness is not None)
        return total_fitness / self.population_size if self.population_size > 0 else None
    
    def get_max_fitness(self) -> float:
        max_fitness = max(ind.fitness for ind in self.individuals if ind.fitness is not None)
        return max_fitness if max_fitness is not None else None
    # other useful metrics for evalution can be added here