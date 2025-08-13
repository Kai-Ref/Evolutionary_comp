from src.TSP import TSP
from src.Local_Search import LocalSearch
from src.operations.mutation.Jump import Jump
from src.operations.mutation.Exchange import Exchange
from src.operations.mutation.TwoOpt import TwoOpt
import argparse

def main():
    # Example call: python main.py -f datasets/eil51.tsp -m local_j -nn 10
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
        choices=['local_j', 'local_e', 'local_i', 'evolutionary'], 
        required=True, 
        help='Type of model to use: local or evolutionary'
    )
    parser.add_argument(
        '--precompute_distances', '-pd',
        action='store_true',
        default=False,
        help='If set, precompute distances (default: False)'
    )
    parser.add_argument(
        '--number_neighbors', '-nn',
        type=int, 
        default=1,
        required=False, 
        help='Number of neighboring mutation combinations to try out'
    )
    parser.add_argument(
        '--max_iterations', '-mi',
        type=int, 
        default=10,
        required=False, 
        help='Maximum number of iterations'
    )
    parser.add_argument(
        '--population_size', '-ps',
        type=int, 
        default=1,
        required=False, 
        help='Size of the population, i.e. number of indpendent individuals'
    )
    args = parser.parse_args()


    if 'local' in args.model_type:
        if args.model_type == 'local_j':
            mutation = Jump()
        elif args.model_type == 'local_e':
            mutation = Exchange()
        elif args.model_type == 'local_i':
            mutation = TwoOpt()
        else:
            raise ValueError("For local search the mutation must be provided.")
        ls = LocalSearch(args.file_path, distance_metric='euclidean', precompute_distances=args.precompute_distances, mutation=mutation, 
                         population_size=args.population_size, number_neighbors=args.number_neighbors)
        ls.solve(max_iterations=args.max_iterations)
    elif args.model_type == 'evolutionary':
        pass
    # tsp = TSP(args.file_path, distance_metric='euclidean', precompute_distances=args.precompute_distances)

if __name__ == "__main__":
    main()