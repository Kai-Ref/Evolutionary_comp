from src.operations.crossover.Crossover import Crossover
from src.Individual import Individual
from typing import override
import numpy as np

class PMX(Crossover):
    #Partially Mixed Crossover is similar to Order crossover in that it tries to keeps an arbitary portion
    #from the parents but if the parents share values in the slice, PMX includes additional values, then
    #copies the rest from the other parent.
    @override
    def xover(self, parent1: Individual, parent2: Individual) -> list:
        added_to_child1 = []
        added_to_child2 = []
        parent_size = len(parent1.permutation)
        #creating a new blank array for the child with a junk value
        child1_tour = np.full(parent_size, np.inf)
        child2_tour = np.full(parent_size, np.inf)
        keep_end = np.random.randint(0, parent_size)

        #randint breaks between (0,0)
        if keep_end == 0:
            keep_start = 0
        else:
            keep_start = np.random.randint(keep_end)

        #copies of parent 1 and 2 to make the crossover a bit easier
        p1_subsequence = []
        p2_subsequence = []

        #steps 2-5 from the slides. it'll take too long to explain it here...
        #this also probably isn't the most efficient way...
        for i in range(keep_start, keep_end + 1):
            child1_tour[i] = parent1.permutation[i] 
            child2_tour[i] = parent2.permutation[i]
            added_to_child1.append(parent1.permutation[i])
            added_to_child2.append(parent2.permutation[i])
            p1_subsequence.append(parent1.permutation[i])
            p2_subsequence.append(parent2.permutation[i])

            for j1 in range(keep_start, keep_end + 1):
                if parent1.permutation[j1] not in p2_subsequence:
                    offset = np.where(parent2.permutation == parent1.permutation[j1])
                    k1 = parent2.permutation[j1]
                    while(k1 in added_to_child1):
                        k1 = parent2.permutation[np.where(parent1.permutation == k1)]
                    child1_tour[offset] = k1
                    added_to_child1.append(k1)

            for j2 in range(keep_start, keep_end + 1):
                if parent2.permutation[j2] not in p1_subsequence:
                    offset = np.where(parent1.permutation == parent2.permutation[j2])
                    k2 = parent1.permutation[j2]
                    while(k2 in added_to_child2):
                        k2 = parent1.permutation[np.where(parent2.permutation == k2)]
                    child2_tour[offset] = k2
                    added_to_child2.append(k2)

        #step 6 from the slides
        for r in range(0, parent_size):
            if child1_tour[r] == np.inf:
                child1_tour[r] = parent2.permutation[r]
            if child2_tour[r] == np.inf:
                child2_tour[r] = parent1.permutation[r]

        child1 = Individual(parent_size, parent1.tsp)
        child1.permutation = child1_tour.tolist()
        child1.fitness += self.efficient_fitness_calculation(child1, parent1, keep_start, keep_end)

        child2 = Individual(parent_size, parent2.tsp)
        child2.permutation = child1_tour.tolist()
        child2.fitness += self.efficient_fitness_calculation(child2, parent2, keep_start, keep_end)        

        return list(child1, child2)


    @override
    def efficient_fitness_calculation(self, individual: Individual, parent: Individual, i: int, j:int) -> float:
        tsp = individual.tsp
        old_tour = parent.permutation
        new_tour = individual.permutation
        n = len(new_tour)
        difference = 0

        #if our copied section is the parent, don't need to calculate
        if(i == 0 and j==(n-1)):
            return 0.0

        #re-calculate every edge except for edges that were shared
        for e in range(n-1):
            if((e <= i) and (e >= j)):
                old_distance = tsp.distance(old_tour[e], old_tour[(e+1)%n])
                new_distance = tsp.distance(new_tour[e], new_tour[(e+1)%n])
                difference += (new_distance - old_distance)

        return difference
    