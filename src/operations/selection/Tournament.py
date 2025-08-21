from src.operations.selection.Selection import Selection
from src.Population import Population
from typing import override
import random


class Tournament(Selection):
    @override
    def __call__(self, population: Population, num_to_select: int) -> Population:
        new_population = Population(population_size=num_to_select,
                                    number_of_nodes=population.individuals[0].permutation.size)

        for _ in range(num_to_select):
            competitors = random.sample(population.individuals, 2)

            winner = max(competitors, key=lambda ind: ind.fitness)
            new_population.individuals.append(winner)

        return new_population
