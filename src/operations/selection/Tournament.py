from src.operations.selection.Selection import Selection
from src.Population import Population
from typing import override
import random


class Tournament(Selection):
    @override
    def __call__(self, population: Population, num_to_select: int) -> Population:
        # Create a new population for the selected individuals
        new_population = Population(population_size=num_to_select,
                                    number_of_nodes=len(population.individuals[0].permutation), tsp=population.individuals[0].tsp)

        # Repeat the tournament to select the number of individuals
        for _ in range(num_to_select):
            # Select two random competitors
            competitors = random.sample(population.individuals, 2)

            # Winner is the individual with the highest fitness
            winner = max(competitors, key=lambda ind: ind.fitness)
            new_population.individuals.append(winner)

        return new_population
