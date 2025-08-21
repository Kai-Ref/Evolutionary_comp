from src.operations.crossover.Crossover import Crossover
from src.Individual import Individual
from typing import override
import numpy as np

class Cycle(Crossover):
    @override
    def xover(self, parent1: Individual, parent2: Individual) -> None:
        added_to_child = []
        parent_size = parent1.permutation.size()
        #creating a new blank array for the child with a junk value
        child1_tour = np.full(parent_size, np.inf)
        child2_tour = np.full(parent_size, np.inf)

        loop_idx = 0
        num_loops = 0
        #main loop: compare parent 1 and 2 at the same index, if they're different, new index is position in p1 where p2's number is
        while(len(added_to_child) != parent_size):
            current_idx = loop_idx
            cycle_start = parent1.permutation[current_idx]
            while (cycle_start != parent2.permutation[current_idx]):
                #every other loop adds to the other parent
                if (num_loops % 2 == 0):
                    child1_tour[current_idx] = parent1.permutation[current_idx]
                    child2_tour[current_idx] = parent2.permutation[current_idx]
                else:
                    child1_tour[current_idx] = parent2.permutation[current_idx]
                    child2_tour[current_idx] = parent1.permutation[current_idx]
                added_to_child.append(parent1.permutation[current_idx])
                #idx in p1 of p2 town
                current_idx = np.where(parent1.permutation == (parent2.permutation[current_idx]))
            
            #add end of cycle
            if (num_loops % 2 == 0):
                child1_tour[current_idx] = parent1.permutation[current_idx]
                child2_tour[current_idx] = parent2.permutation[current_idx]
            else:
                child1_tour[current_idx] = parent2.permutation[current_idx]
                child2_tour[current_idx] = parent1.permutation[current_idx]
            added_to_child.append(parent1.permutation[current_idx])
            current_idx = np.where(parent1.permutation == (parent2.permutation[current_idx]))
            
            #find the next cycle
            for i in range(loop_idx, parent_size):
                if parent1.permutation[i] not in added_to_child:
                    loop_idx = i
                    break
            num_loops += 1
        
        child1 = Individual(parent_size, parent1.tsp)
        child1.permutation = child1_tour.tolist()
        child1.fitness = self.efficient_fitness_calculation()

        child2 = Individual(parent_size, parent2.tsp)
        child2.permutation = child1_tour.tolist()
        child2.fitness = self.efficient_fitness_calculation()
        return tuple(child1, child2)
        



    @override
    def efficient_fitness_calculation(self, individual: Individual) -> None:
        raise NotImplementedError("Efficient fitness calculation for %1 is not implemented yet.".format(self.__class__.__name__))
    