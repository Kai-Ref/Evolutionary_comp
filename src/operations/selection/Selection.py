from src.Population import Population
from src.Individual import Individual
class Selection:
    def __call__(self, population: Population) -> Population:
        raise NotImplementedError("Selection method must be implemented in subclasses.")
    