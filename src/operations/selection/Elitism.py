from src.operations.selection.Selection import Selection
from src.Population import Population
from typing import override


class Elitism(Selection):
    @override
    def __call__(self, population: Population, num_to_select: int) -> Population:
        population.individuals.sort(key=lambda x: x.fitness)
        selected_individuals = population.individuals[:num_to_select]
        new_population = Population(population_size=len(selected_individuals),
                                    number_of_nodes=selected_individuals[0].permutation.size)
        new_population.population_size = num_to_select
        new_population.individuals = selected_individuals

        return new_population
