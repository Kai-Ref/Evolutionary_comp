from src.operations.selection.Selection import Selection
from src.Population import Population
from typing import override
import numpy


class FitnessBased(Selection):
    @override
    def __call__(self, population: Population, num_to_select: int) -> Population:
        # Find the total sum of fitness values in the population
        total_fitness = sum(individual.fitness for individual in population.individuals)

        # Calculate the selection probabilities based on fitness
        probabilities = [individual.fitness / total_fitness for individual in population.individuals]

        # Choose num_to_select individuals based on their fitness probabilities
        selected_individuals = numpy.random.choice(
            population.individuals,
            size=num_to_select,
            p=probabilities
        ).tolist()

        # Create a new population with the selected individuals
        new_population = Population(population_size=num_to_select,
                                    number_of_nodes=len(selected_individuals[0].permutation), tsp=selected_individuals[0].tsp)

        new_population.individuals = selected_individuals

        # Return the new population
        return new_population
