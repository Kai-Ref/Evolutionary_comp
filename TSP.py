import numpy as np
from scipy.spatial.distance import cdist

class TSP:
    def __init__(self, filepath: str, distance_metric: str = 'euclidean', precompute_distances: bool = True):
        self.filepath = filepath
        self.metadata = {}
        self.node_coords = None  # Will be a NumPy array
        self.distance_metric = distance_metric
        self.precompute_distances = precompute_distances
        self.distance_matrix = None

    def read(self) -> None:
        coords_list = []
        with open(self.filepath, 'r') as file:
            lines = file.readlines()

        in_node_section = False
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line == "NODE_COORD_SECTION":
                in_node_section = True
                continue
            elif line == "EOF":
                break

            if in_node_section:
                parts = line.split()
                if len(parts) >= 3:
                    x = float(parts[1])
                    y = float(parts[2])
                    coords_list.append((x, y))
            else:
                if ':' in line:
                    key, value = line.split(':', 1)
                    self.metadata[key.strip()] = value.strip()

        self.node_coords = np.array(coords_list)

        if self.precompute_distances:
            self.distance_matrix = cdist(self.node_coords, self.node_coords, metric=self.distance_metric)

    def get_metadata(self) -> dict:
        return self.metadata

    def get_node_coords(self) -> np.ndarray:
        return self.node_coords

    def get_distance_matrix(self) -> np.ndarray:
        if self.distance_matrix is None:
            self.distance_matrix = cdist(self.node_coords, self.node_coords, metric=self.distance_metric)
        return self.distance_matrix

    def distance(self, i: int, j: int) -> float:
        """Compute distance between node i and j (0-based index)."""
        if self.distance_matrix is not None:
            return self.distance_matrix[i, j]
        else:
            return cdist([self.node_coords[i]], [self.node_coords[j]], metric=self.distance_metric)[0, 0]

    def __repr__(self) -> str:
        n = len(self.node_coords) if self.node_coords is not None else 0
        return f"<TSP {self.filepath}, {n} nodes, metric={self.distance_metric}>"
    
if __name__ == "__main__":
    tsp = TSP('datasets/eil51.tsp', distance_metric='euclidean', precompute_distances=True)
    tsp.read()
    print(tsp.get_metadata())
    print(tsp.get_node_coords().shape)
    print("Distance between node 0 and 1:", tsp.distance(0, 1))