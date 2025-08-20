from src.operations.crossover.Crossover import Crossover
from src.Individual import Individual
from typing import override
import numpy as np

class EdgeRecombination(Crossover):
    @override
    def xover(self, parent1: Individual, parent2: Individual) -> None:
            #TODO: continue writing this
            child = np.full(parent_size, np.inf)

            parent_size = 4
            edge_table = np.zeros((parent_size, parent_size))
            #create edge table
            for i in range(0, parent_size):
                edge_table[parent1[i]][parent1[i-1]] += 1
                edge_table[parent1[i]][parent1[(i+1)%parent_size]] += 1
                edge_table[parent2[i]][parent2[i-1]] += 1
                edge_table[parent2[i]][parent2[(i+1)%parent_size]] += 1
            
            node = np.random.randint(0, parent_size-1)
            choices = []
            free = 0

            while len(child) < parent_size:
                choices.append(node)
                for e in edge_table:
                    edge_table[e][node] = 0

                if (np.sum(edge_table[node]) > 0):
                    adj_edges = np.where(choices == 2)

    @override
    def efficient_fitness_calculation(self, individual: Individual) -> None:
        raise NotImplementedError("Efficient fitness calculation for %1 is not implemented yet.".format(self.__class__.__name__))
    