from src.operations.crossover.Crossover import Crossover
from src.Individual import Individual
from typing import override
import numpy as np

class Order(Crossover):
    #Order Crossover takes two parent Individuals as an arguement and returns ONE single child.
    #As per the lecture, the child retains a random sequence from the first parent and fills the rest
    #of the aleles from the second parent. 
    @override
    def xover(self, parent1: Individual, parent2: Individual) -> Individual:
        added_to_child_1 = []
        added_to_child_2 = []
        parent_size = parent1.permutation.size()
        child1 = np.full(parent_size, np.inf)
        child2 = np.full(parent_size, np.inf)

        #keep_start and keep_end are the start and end of the random selection from parent 1
        keep_end = np.random.randint(0, parent_size)
        #randint breaks between (0,0)
        if keep_end == 0:
            keep_start = 0
        else:
            keep_start = np.random.randint(keep_end)

        #insertion from parent1 to child
        for i in range(keep_start, keep_end + 1):
            child1[i] = parent1.permutation[i]
            child2[i] = parent2.permutation[i] 
            added_to_child_1.append(parent1.permutation[i])
            added_to_child_2.append(parent2.permutation[i])

        keep_end += 1
        #c_free is the next available free slot in child
        c1_free, c2_free = keep_end, keep_end

        #loop throught the second parent to fill in gaps
        while (np.inf in child1):
            if keep_end >= parent_size:
                keep_end = 0

            if c1_free >= parent_size:
                c1_free = 0

            if (parent2.permutation[keep_end] not in added_to_child_1) and (child1[c1_free] == np.inf):
                child1[c1_free] = parent2[keep_end]
                added_to_child_1.append(parent2[keep_end])
                c1_free += 1

            if (parent1.permutation[keep_end] not in added_to_child_2) and (child2[c2_free] == np.inf):
                child1[c1_free] = parent1.permutation[keep_end]
                added_to_child_2.append(parent1.permutation[keep_end])
                c1_free += 1

            keep_end += 1

        #TODO: calculate fitness here

        return tuple(child1, child2)
   
    @override
    def efficient_fitness_calculation(self, individual: Individual) -> None:
        raise NotImplementedError("Efficient fitness calculation for %1 is not implemented yet.".format(self.__class__.__name__))
    