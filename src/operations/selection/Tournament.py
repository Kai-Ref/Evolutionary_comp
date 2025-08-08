from src.operations.selection.Selection import Selection
from src.Population import Population
from typing import override
class Tournament(Selection):
    @override
    def __call__(self, population: Population) -> Population:
        raise NotImplementedError(f"{self.__class__.__name__} is not implemented yet.")
