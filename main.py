from src.TSP import TSP

if __name__ == "__main__":
    tsp = TSP('datasets/eil51.tsp', distance_metric='euclidean', precompute_distances=True)
    print(tsp.get_metadata())
    print(tsp.get_node_coords().shape)
    print("Distance between node 0 and 1:", tsp.distance(0, 1))