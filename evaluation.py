import numpy as np

algorithms = ['Jump', 'Exchange', 'TwoOpt']
datasets = ['eil51', 'eil76', 'eil101', "st70", "kroA100", "kroC100", "kroD100", "lin105", "pcb442", \
            "pr2392", "usa13509"]

def evaluate():
    results = []
    for dataset in datasets:
        for alg in algorithms:
            last_values = []
            for i in range(1, 31):
                file_path = f"data/{dataset}/{alg}/Individual_{i}_fitness_history_individual.npy"
                
                try:
                    arr = np.load(file_path)
                    last_values.append(arr[-1])
                except Exception as e:
                    print(f"Could not load {file_path}: {e}")

            if last_values:
                min_val = np.min(last_values)
                mean_val = np.mean(last_values)
                results.append((alg, dataset, min_val, mean_val))
                print(f"{alg:9} | {dataset:10} | Min: {min_val:.2f} | Mean: {mean_val:.2f}")


if __name__ == "__main__":
    evaluate()