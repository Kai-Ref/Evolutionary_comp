from src.operations.selection.Selection import Selection
from src.Population import Population
from typing import override


class Elitism(Selection):
    @override
    def __call__(self, population: Population, num_to_select: int) -> Population:
        # Sort the individuals based on their fitness
        population.individuals.sort(key=lambda x: x.fitness)

        # Select the top individuals based on the specified number to select
        selected_individuals = population.individuals[:num_to_select]

        # Create a new population with the selected individuals
        new_population = Population(population_size=len(selected_individuals),
                                    number_of_nodes=selected_individuals[0].permutation.size)

        new_population.individuals = selected_individuals

        # Return the new population
        return new_population
