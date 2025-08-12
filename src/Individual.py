import numpy as np
import copy
# from src.TSP import TSP

class Individual:
    permutation: np.ndarray
    fitness: float
    
    def __init__(self, number_of_nodes: int, tsp = None):
        self.permutation = np.random.permutation(number_of_nodes).tolist()
        self.tsp = tsp
        self.is_local_optimum = False
        self.calculate_fitness()

    def copy(self):
        return copy.deepcopy(self)

    def __str__(self):
        return f"Individual(permutation={self.permutation}, fitness={self.fitness})"

    def __repr__(self):
        return self.__str__()
    
    def calculate_fitness(self) -> None:
        self.fitness = 0
        for i in range(len(self.permutation) - 1):
            self.fitness += self.tsp.distance(self.permutation[i], self.permutation[i + 1])
        self.fitness += self.tsp.distance(self.permutation[-1], self.permutation[0])
