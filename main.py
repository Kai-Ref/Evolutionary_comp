# EA Mods start @Shen
# Import modules necessary for local search are moved inside the if block 
from src.EvolutionaryAlgorithm import EvolutionaryAlgorithm
from src.operations.selection.Tournament import Tournament
from src.operations.selection.FitnessBased import FitnessBased
from src.operations.crossover.Order import Order
from src.operations.mutation.Exchange import Exchange
# EA Mods end @Shen


import argparse
# EA Mods start @Shen
import random
import numpy as np
# EA Mods end @Shen

def main():
    # Example call: python main.py -f datasets/eil51.tsp -m local_j -pd -mi 1000 -ps 1
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
    # EA Mods start: EA args @Shen
    parser.add_argument('--ea_generations','-eg', type=int, default=2000)
    parser.add_argument('--ea_selection','-es', type=str, default='tournament', choices=['tournament','fitness'])
    parser.add_argument('--ea_crossover_rate','-cr', type=float, default=0.9)
    parser.add_argument('--ea_mutation_rate','-mr', type=float, default=0.2)
    parser.add_argument('--ea_elitism_k','-ek', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    # EA Mods end: EA args @Shen

    args = parser.parse_args()

    if args.model_type.startswith('local_'):
        # EA Mods start @Shen
        from src.Local_Search import LocalSearch
        from src.operations.mutation.Jump import Jump
        from src.operations.mutation.Exchange import Exchange as LSExchange
        from src.operations.mutation.TwoOpt import TwoOpt
        # EA Mods end @Shen
        
        if args.model_type == 'local_j':
            mutation = Jump()
        elif args.model_type == 'local_e':
            mutation = LSExchange()
        elif args.model_type == 'local_i':
            mutation = TwoOpt()
        else:
            raise ValueError("Unknown local model type.")

        ls = LocalSearch(
            args.file_path,
            distance_metric='euclidean',
            precompute_distances=args.precompute_distances,
            mutation=mutation,
            population_size=args.population_size
        )
        ls.solve(max_iterations=args.max_iterations)
        

    # EA Mods start @Shen 

    ### Evolutionary Variant_A

    elif args.model_type == 'evolutionary':
            
            rng = random.Random(args.seed)
            np.random.seed(args.seed)

            selection = Tournament(k=3, rng=rng) if args.ea_selection == 'tournament' else FitnessBased(rng=rng)
            crossover = Order(rng=rng)
            mutation = Exchange()

            ea = EvolutionaryAlgorithm(
                filepath=args.file_path,
                population_size=args.population_size,
                precompute_distances=args.precompute_distances,
                selection=selection,
                crossover=crossover,
                mutation=mutation,
                crossover_rate=args.ea_crossover_rate,
                mutation_rate=args.ea_mutation_rate,
                elitism_k=args.ea_elitism_k,
                seed=args.seed,
                log_dir="results/ea_variant_a"
            )
            best = ea.solve(max_generations=args.ea_generations)
            print(f"Best tour cost: {best.fitness}")
            print(f"Best permutation: {best.permutation}")
            
    # EA Mods end @Shen


if __name__ == "__main__":
    main()