#!/bin/bash
#SBATCH -p cpu_il  # cpu_il, cpu or highmem
#SBATCH -n 1                   # Number of tasks (1 for single node)
#SBATCH -t 500           # Time limit (10 minutes for debugging purposes)
#SBATCH --mem=70000             # Memory request (adjust as needed)
#SBATCH --cpus-per-task=16     # Number of CPUs per GPU (16 for A100)
#SBATCH --ntasks-per-node=1    # Number of tasks per node (1 in this case)

module load devel/miniforge

conda activate evo

# Example: list of datasets
# datasets=("kroC100.tsp" "lin105.tsp" "kroD100.tsp" "st70.tsp" "pcb442.tsp") # "eil76.tsp" "eil101.tsp" "kroA100.tsp") ("pr2392.tsp")
# datasets=("pr2392.tsp")
datasets=("usa13509.tsp")
# algos=("local_j" "local_e" "local_i")
algos=("local_i")
# algos=("local_e")
# algos=("local_j")

for ds in "${datasets[@]}"; do
    echo "Running dataset: $ds"
    for algo in "${algos[@]}"; do
        for i in {1..30}; do
            python main.py -f "datasets/$ds" -m "$algo" -nn -1 -pd -mi 1000 -ps 1
        done
    done
done