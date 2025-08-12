from src.TSP import TSP
import argparse

def main():
    # Parsing arguments: File path and model type 
    parser = argparse.ArgumentParser(description="Run TSP solver")
    parser.add_argument(
        '--file_path', '-f', 
        type=str, 
        required=True, 
        help='Path to the TSP dataset file'
    )
    parser.add_argument(
        '--model_type', '-m', 
        type=str, 
        choices=['local', 'evolutionary'], 
        required=True, 
        help='Type of model to use: local or evolutionary'
    )
    parser.add_argument(
        '--precompute_distances', '-pd',
        action='store_true',
        default=False,
        help='If set, precompute distances (default: False)'
    )
    args = parser.parse_args()

    tsp = TSP(args.file_path, distance_metric='euclidean', precompute_distances=args.precompute_distances)
    print(tsp.get_metadata())
    print(tsp.get_node_coords().shape)
    print("Distance between node 0 and 1:", tsp.distance(0, 1))

    print(f"Selected model type: {args.model_type}")

if __name__ == "__main__":
    main()