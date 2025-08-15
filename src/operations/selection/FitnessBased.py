from src.operations.selection.Selection import Selection
from src.Population import Population
from typing import override
import numpy


class FitnessBased(Selection):
    @override
    def __call__(self, population: Population, num_to_select: int) -> Population:
        total_fitness = sum(individual.fitness for individual in population.individuals)

        probabilities = [individual.fitness / total_fitness for individual in population.individuals]

        selected_individuals = numpy.random.choice(
            population.individuals,
            size=num_to_select,
            p=probabilities
        ).tolist()

        new_population = Population(population_size=num_to_select,
                                    number_of_nodes=selected_individuals[0].permutation.size)

        new_population.individuals = selected_individuals

        return new_population
