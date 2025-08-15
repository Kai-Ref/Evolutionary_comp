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
        c_added = []
        parent_size = parent1.permutation.size()
        child = np.full(parent_size, np.inf)

        #keep_start and keep_end are the start and end of the random selection from parent 1
        keep_end = np.random.randint(0, parent_size)
        #randint breaks between (0,0)
        if keep_end == 0:
            keep_start = 0
        else:
            keep_start = np.random.randint(keep_end)

        #insertion from parent1 to child
        for i in range(keep_start, keep_end + 1):
            child[i] = parent1.permutation[i] 
            c_added.insert(-1, parent1.permutation[i])

        keep_end += 1
        #c_free is the next available free slot in child
        c_free = keep_end

        #loop throught the second parent in 
        while (np.inf in child):
            if keep_end >= parent_size:
                keep_end = 0

            if c_free >= parent_size:
                c_free = 0

            if (parent2[keep_end] not in c_added) and (child[c_free] == np.inf):
                child[c_free] = parent2[keep_end]
                c_added.insert(-1, parent2[keep_end])
                c_free += 1

            keep_end += 1

        return child
   
    @override
    def efficient_fitness_calculation(self, individual: Individual) -> None:
        raise NotImplementedError("Efficient fitness calculation for %1 is not implemented yet.".format(self.__class__.__name__))
    