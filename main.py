from src.Local_Search import LocalSearch
from src.operations.mutation.Jump import Jump

if __name__ == "__main__":
    mutation = Jump()
    tsp = LocalSearch('datasets/eil51.tsp', distance_metric='euclidean', precompute_distances=True, population_size=1, mutation=mutation)
    tsp.solve(max_iterations=2)
    print(tsp.get_metadata())
    print(tsp.get_node_coords().shape)
    print("Distance between node 0 and 1:", tsp.distance(0, 1))