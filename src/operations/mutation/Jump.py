from src.operations.mutation.Mutation import Mutation
from src.Individual import Individual
from typing import override
import numpy as np

class Jump(Mutation):
    @override
    def mutate_individual(self, individual: Individual, i: int, j: int, update_fitness: bool = False) -> Individual:
        n = len(individual.permutation)
        assert 0 <= i < n, "Index i is out of bounds."
        assert 0 <= j < n, "Index j is out of bounds."
        assert i != j, "Indices i and j must be different."
        
        # place i in position j and shift rest
        new_tour = individual.permutation.copy()  # Copy the original tour
        moved_city = new_tour.pop(i)
        new_tour.insert(j, moved_city)

        # Create a new individual and compute its fitness efficiently
        new_individual = Individual(permutation=new_tour, tsp=individual.tsp)
        # if update_fitness as this might not be required for EA
        new_individual.fitness = self.efficient_fitness_calculation(individual, i, j) if update_fitness else None
        return new_individual
    
    @override
    def efficient_fitness_calculation(self, individual, i, j):
        n = len(individual.permutation)
        tsp = individual.tsp
            
        i_prev = (i - 1) % n
        i_next = (i + 1) % n
        j_prev = (j - 1) % n
        j_next = (j + 1) % n


        if i > j:
            old_distance = tsp.distance(individual.permutation[i_prev], individual.permutation[i]) + \
                        tsp.distance(individual.permutation[i], individual.permutation[i_next]) + \
                        tsp.distance(individual.permutation[j_prev], individual.permutation[j])
                        
            new_distance = tsp.distance(individual.permutation[i_prev], individual.permutation[i_next]) + \
                        tsp.distance(individual.permutation[j_prev], individual.permutation[i]) + \
                        tsp.distance(individual.permutation[i], individual.permutation[j])
        else:
            old_distance = tsp.distance(individual.permutation[i_prev], individual.permutation[i]) + \
                        tsp.distance(individual.permutation[i], individual.permutation[i_next]) + \
                        tsp.distance(individual.permutation[j], individual.permutation[j_next])
                        
            new_distance = tsp.distance(individual.permutation[i_prev], individual.permutation[i_next]) + \
                        tsp.distance(individual.permutation[j_next], individual.permutation[i]) + \
                        tsp.distance(individual.permutation[j], individual.permutation[i])

        return individual.fitness + (new_distance - old_distance)
