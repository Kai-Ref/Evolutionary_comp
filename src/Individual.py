import numpy as np

class Individual:
    permutation: np.ndarray
    fitness: float
    
    def __init__(self, number_of_nodes: int):
        self.permutation = np.random.permutation(number_of_nodes)
        self.fitness = None
        self.is_local_optimum = False
