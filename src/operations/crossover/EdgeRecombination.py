from src.operations.crossover.Crossover import Crossover
from src.Individual import Individual
from typing import override
import numpy as np

class EdgeRecombination(Crossover):
    @override
    def xover(self, parent1: Individual, parent2: Individual) -> Individual:
        """
        Edge recombination takes the edges from the parents, adds them to a table, and reconstructs a child
        via a set of rules. It'll first pick a common edge, then the smallest child duplicate, and finally
        one and random.
        """
        parent_size = len(parent1.permutation)
        child_tour = np.full(parent_size, np.inf)
        edge_table = np.zeros((parent_size, parent_size), dtype=int)
        #create edge table
        for i in range(0, parent_size):
            edge_table[parent1.permutation[i]][parent1.permutation[i-1]] += 1
            edge_table[parent1.permutation[i]][parent1.permutation[(i+1)%parent_size]] += 1
            edge_table[parent2.permutation[i]][parent2.permutation[i-1]] += 1
            edge_table[parent2.permutation[i]][parent2.permutation[(i+1)%parent_size]] += 1
        
        node = np.random.randint(0, parent_size-1)
        next_free_in_child = 0
        remaining_nodes = list(range(0, parent_size))

        #start the edge recombination
        while len(remaining_nodes) > 0:
            #add node to child
            child_tour[next_free_in_child] = node
            next_free_in_child += 1

            #remove it from available and the table
            remaining_nodes.remove(node)
            for e in range(0, parent_size):
                edge_table[e][node] = 0
                edge_table[node][e] = 0

            #check if current node has anything related to it
            if (np.sum(edge_table[node]) > 0):
                adj_edges = np.where(edge_table[node] == 2)
                #first: pick adjectent edges (edge with value 2)
                if(len(adj_edges[0])):
                    node = adj_edges[0][0]
                else:
                    #then find the node with the smallest edges
                    min_node = 999
                    for j in edge_table[node]:
                        j_sum = np.sum(edge_table[j])
                        if(j_sum <= min_node):
                            #if it's the smallest edge
                            min_node = j_sum
                            node = j

            else:
                #otherwise it's a random node that hasn't been selected yet
                if(np.inf in child_tour):
                    node = np.random.choice(remaining_nodes, 1)[0]

        child = Individual(parent_size, parent1.tsp)
        child.permutation = child_tour.astype(int).tolist()
        child.fitness = self.efficient_fitness_calculation(child)
        return child

    @override
    def efficient_fitness_calculation(self, individual: Individual) -> float:
        tsp = individual.tsp
        new_tour = individual.permutation
        n = len(new_tour)
        difference = 0

        for e in range(n):
            distance = tsp.distance(new_tour[e], new_tour[(e+1)%n])
            difference += distance

        return difference
    