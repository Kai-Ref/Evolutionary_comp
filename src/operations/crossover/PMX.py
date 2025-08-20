from src.operations.crossover.Crossover import Crossover
from src.Individual import Individual
from typing import override
import numpy as np

class PMX(Crossover):
    #Partially Mixed Crossover is similar to Order crossover in that it tries to keeps an arbitary portion
    #from the parents but if the parents share values in the slice, PMX includes additional values, then
    #copies the rest from the other parent.
    @override
    def xover(self, parent1: Individual, parent2: Individual) -> Individual:
        added_to_child = []
        parent_size = parent1.permutation.size()
        #creating a new blank array for the child with a junk value
        child1 = np.full(parent_size, np.inf)
        child2 = np.full(parent_size, np.inf)
        keep_end = np.random.randint(0, parent_size)

        #randint breaks between (0,0)
        if keep_end == 0:
            keep_start = 0
        else:
            keep_start = np.random.randint(keep_end)

        #copies of parent 1 and 2 to make the crossover a bit easier
        p2_subsequence = []

        #steps 2-5 from the slides. it'll take too long to explain it here...
        #this also probably isn't the most efficient way...
        for i in range(keep_start, keep_end + 1):
            child1[i] = parent1.permutation[i] 
            added_to_child.append(parent1.permutation[i])
            p2_subsequence.append(parent2.permutation[i])

            for j in range(keep_start, keep_end + 1):
                if parent1.permutation[j] not in p2_subsequence:
                    offset = np.where(parent2.permutation == parent1.permutation[j])
                    k = parent2.permutation[j]
                    while(k in added_to_child):
                        k = parent2.permutation[np.where(parent1.permutation == k)]
                    child1[offset] = k
                    added_to_child.insert(-1, k)

        #step 6 from the slides
        for r in range(0, parent_size):
            if child1[r] == np.inf:
                child1[r] = parent2.permutation[r]

        #TODO: now calculate child2
        #TODO: calc fitness
        return child1


    @override
    def efficient_fitness_calculation(self, individual: Individual) -> None:
        raise NotImplementedError("Efficient fitness calculation for %1 is not implemented yet.".format(self.__class__.__name__))
    