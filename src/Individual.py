import numpy as np
import copy

class Individual:
    permutation: np.ndarray
    fitness: float
    
    def __init__(self, number_of_nodes: int):
        self.permutation = np.random.permutation(number_of_nodes)
        self.fitness = None

    def copy(self):
        return copy.deepcopy(self)

    def __str__(self):
        return f"Individual(permutation={self.permutation}, fitness={self.fitness})"

    def __repr__(self):
        return self.__str__()