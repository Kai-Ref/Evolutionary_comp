from __future__ import annotations

import os, sys, argparse, random
import numpy as np
from typing import Optional, Sequence, Tuple, List

from src.TSP import TSP
from src.Population import Population
from src.Individual import Individual
from src.FileWriter import FileWriter

#  Mutation operators 
from src.operations.mutation.TwoOpt import TwoOpt
from src.operations.mutation.Jump import Jump
from src.operations.mutation.Exchange import Exchange

#  Local (minimization) Tournament selection 
class MinTournament:
    """Minimization tournament selection for TSP (lower fitness is better)."""
    def __init__(self, k: int = 3, rng: random.Random | None = None):
        self.k = k
        self.rng = rng or random.Random()

    def __call__(self, population: Population, num_to_select: int) -> Population:
        selected = []
        for _ in range(num_to_select):
            competitors = self.rng.sample(population.individuals, self.k)
            winner = min(competitors, key=lambda ind: ind.fitness)  # MIN
            selected.append(winner)

        n_nodes = len(population.individuals[0].permutation)
        tsp_obj = population.individuals[0].tsp
        new_pop = Population(population_size=num_to_select,
                             number_of_nodes=n_nodes,
                             tsp=tsp_obj)
        new_pop.individuals = selected
        return new_pop


class EvolutionaryAlgorithmC(TSP):
    """
    Variant C: Generational, mutation-only EA with elitism + diversity options 
    - Selection: MinTournament(k=3) by default 
    """

    def __init__(
        self,
        filepath: str,
        population_size: int = 100,
        distance_metric: str = "euclidean",
        precompute_distances: bool = True,
        # Operators
        selection: Optional[object] = None,               # default: MinTournament(k=3)
        mutation_ops: Optional[Sequence[object]] = None,  # default: [TwoOpt, Jump, Exchange]
        # Hyperparameters
        mutation_rate: float = 1.0,    # mutate every child (mutation-only)
        elitism_k: int = 2,
        seed: Optional[int] = None,
        log_dir: str = "results/ea_variant_c",
        # mutation scheduling
        mutation_weights: Optional[Tuple[float, float, float]] = None,        # (TwoOpt, Jump, Exchange)
        mutation_strength_probs: Optional[Tuple[float, float, float]] = None, # P(apply 1/2/3 mutations)
        # optional diversity / memetic
        immigrant_rate: float = 0.10,      # replace worst r% as immigrants
        immigrant_period: int = 100,       # every P generations
        elite_memetic_frac: float = 0.10,  # fraction polished
        memetic_budget: int = 10,          # 2-opt attempts per elite
    ):
        super().__init__(
            filepath=filepath,
            distance_metric=distance_metric,
            precompute_distances=precompute_distances,
            population_size=population_size,
            mutation=None,
        )

        # RNG
        self.seed = seed
        self.rng = random.Random(seed)
        np.random.seed(seed if seed is not None else None)

        # Selection (local, minimization)
        self.selection = selection or MinTournament(k=3, rng=self.rng)

        # Mutation ops
        self.mutation_ops = list(mutation_ops) if mutation_ops is not None else [TwoOpt(), Jump(), Exchange()]
        self.op_names = [str(op) for op in self.mutation_ops]
        self.two_opt = TwoOpt()  # used for memetic polish

        # Size-aware schedules from LS findings
        n_nodes = len(self.population.individuals[0].permutation)
        self.mutation_weights = mutation_weights or self.size_aware_weights(n_nodes)
        self.mutation_strength_probs = mutation_strength_probs or self.size_aware_strength_probability(n_nodes)

        # Hyperparams
        self.mutation_rate = float(mutation_rate)
        self.elitism_k = int(elitism_k)

        # Diversity / memetic
        self.immigrant_rate = float(immigrant_rate)
        self.immigrant_period = int(immigrant_period)
        self.elite_memetic_frac = float(elite_memetic_frac)
        self.memetic_budget = int(memetic_budget)

        # Logging
        self.file_writer = FileWriter()
        self.log_dir_base = log_dir

        # Ensure initial fitness exists
        for ind in self.population.individuals:
            if ind.fitness is None:
                ind.calculate_fitness()

    # size-aware schedules 
    def size_aware_weights(self, n: int) -> Tuple[float, float, float]:
        # (TwoOpt, Jump, Exchange)
        if n <= 100:
            return (0.65, 0.25, 0.10)
        elif n <= 500:
            return (0.55, 0.25, 0.20)
        elif n <= 3000:
            return (0.45, 0.25, 0.30)
        else:
            return (0.30, 0.20, 0.50)

    def size_aware_strength_probability(self, n: int) -> Tuple[float, float, float]:
        # Probability of applying 1 / 2 / 3 mutations per child
        if n <= 100:
            return (0.70, 0.20, 0.10)
        elif n <= 500:
            return (0.60, 0.30, 0.10)
        elif n <= 3000:
            return (0.50, 0.30, 0.20)
        else:
            return (0.40, 0.35, 0.25)

    # Helper Functions
    def elitism(self, pop: Population, k: int) -> list[Individual]:
        return sorted(pop.individuals, key=lambda ind: ind.fitness)[:max(0, k)]

    def select_parents(self, pop: Population, n_pairs: int) -> list[tuple[Individual, Individual]]:
        pairs = []
        for _ in range(n_pairs):
            sel = self.selection(pop, 2)
            p1, p2 = sel.individuals[0], sel.individuals[1]
            pairs.append((p1, p2))
        return pairs

    def clone(self, ind: Individual) -> Individual:
        c = Individual(number_of_nodes=None, tsp=ind.tsp, permutation=ind.permutation.copy())
        c.fitness = ind.fitness
        return c

    def apply_one_mutation(self, child: Individual) -> Individual:
        op = self.rng.choices(self.mutation_ops, weights=self.mutation_weights, k=1)[0]
        n = len(child.permutation)
        i, j = self.rng.sample(range(n), 2)
        if isinstance(op, TwoOpt) and i > j:
            i, j = j, i
        mutated = op.mutate_individual(child, i, j, update_fitness=True)
        if mutated.fitness is None:
            mutated.calculate_fitness()
        return mutated

    def maybe_mutate(self, child: Individual) -> Individual:
        if self.rng.random() >= self.mutation_rate:
            return child
        strength = self.rng.choices([1, 2, 3], weights=self.mutation_strength_probs, k=1)[0]
        for _ in range(strength):
            child = self.apply_one_mutation(child)
        return child

    def memetic_polish(self, ind: Individual, attempts: int) -> Individual:
        cur = ind
        for _ in range(attempts):
            n = len(cur.permutation)
            i, j = sorted(self.rng.sample(range(n), 2))
            cand = self.two_opt.mutate_individual(cur, i, j, update_fitness=True)
            if cand.fitness < cur.fitness:
                cur = cand
        return cur

    def inject_immigrants(self, pop: Population, frac: float) -> None:
        m = max(1, int(frac * len(pop.individuals)))
        pop.individuals.sort(key=lambda ind: ind.fitness)  # ascending
        worst_idxs = list(range(len(pop.individuals) - m, len(pop.individuals)))
        n_nodes = len(pop.individuals[0].permutation)
        for idx in worst_idxs:
            newcomer = Individual(number_of_nodes=n_nodes, tsp=self)  # random perm; fitness computed in ctor
            pop.individuals[idx] = newcomer

    # Testing 
    def log_arrays(self, instance_name: str, seed: Optional[int], best_hist: List[float], mean_hist: List[float]):
        pop_n = len(self.population.individuals)
        folder = os.path.join(self.log_dir_base, instance_name, f"pop_{pop_n}", f"seed_{seed}")
        os.makedirs(folder, exist_ok=True)
        self.file_writer.file_path = folder
        self.file_writer(np.array(best_hist), "best_cost_per_generation.npy")
        self.file_writer(np.array(mean_hist), "mean_cost_per_generation.npy")

    # public API
    def solve(self, max_generations: int = 20000) -> Individual:
        pop = self.population
        n = len(pop.individuals)
        instance_name = os.path.splitext(os.path.basename(self.filepath))[0]

        best_history: List[float] = []
        mean_history: List[float] = []

        best = min(pop.individuals, key=lambda ind: ind.fitness)
        best_history.append(best.fitness)
        mean_history.append(sum(ind.fitness for ind in pop.individuals) / n)

        for gen in range(int(max_generations)):
            elites = self.elitism(pop, self.elitism_k)

            n_pairs = (n - len(elites)) // 2
            next_inds: List[Individual] = elites.copy()

            for (p1, p2) in self.select_parents(pop, n_pairs):
                c1 = self.maybe_mutate(self.clone(p1))
                c2 = self.maybe_mutate(self.clone(p2))
                next_inds.extend([c1, c2])

            while len(next_inds) < n:
                lone = self.selection(pop, 1).individuals[0]
                c = self.maybe_mutate(self.clone(lone))
                next_inds.append(c)

            # memetic polish on top slice
            polish_k = max(1, int(self.elite_memetic_frac * n))
            next_inds.sort(key=lambda ind: ind.fitness)
            for i in range(polish_k):
                next_inds[i] = self.memetic_polish(next_inds[i], attempts=self.memetic_budget)

            # diversity injection every P generations
            if self.immigrant_period > 0 and (gen + 1) % self.immigrant_period == 0:
                tmp = Population(population_size=n, number_of_nodes=len(next_inds[0].permutation), tsp=self)
                tmp.individuals = next_inds
                self.inject_immigrants(tmp, self.immigrant_rate)
                next_inds = tmp.individuals

            pop.individuals = next_inds[:n]
            pop.population_size = n

            cur_best = min(pop.individuals, key=lambda ind: ind.fitness)
            if cur_best.fitness < best.fitness:
                best = cur_best
            best_history.append(best.fitness)
            mean_history.append(sum(ind.fitness for ind in pop.individuals) / n)

        self.log_arrays(instance_name, self.seed, best_history, mean_history)
        return best

# Testing 
DATASETS = [
    "eil51","eil76", "st70", "kroA100", "kroC100", "kroD100", "lin105", "pcb442", \
            "pr2392", "usa13509"
]
POPS = [20, 50, 100, 200]
VARIANT_DIR = "results/ea_variant_c"
OUT_TXT = "results/EA_mutation_results.txt"

def curve_path(inst: str, pop: int, seed: int) -> str:
    return os.path.join(VARIANT_DIR, inst, f"pop_{pop}", f"seed_{seed}", "best_cost_per_generation.npy")

def ensure_run(inst: str, pop: int, seed: int, gens: int) -> None:
    """
    Run EA-C for (inst, pop, seed) if curve is missing or too short.
    """
    fp = curve_path(inst, pop, seed)
    try:
        arr = np.load(fp)
        if len(arr) >= gens + 1:
            return
    except Exception:
        pass

    ds_path = os.path.join("datasets", f"{inst}.tsp")
    ea = EvolutionaryAlgorithmC(filepath=ds_path,
                                population_size=pop,
                                precompute_distances=True,
                                seed=seed,
                                log_dir=VARIANT_DIR)
    ea.solve(max_generations=gens)

def read_checkpoints(inst: str, pop: int, seeds: int, checkpoints: List[int], gens: int, reducer: str):
    
    rows, used = [], 0
    for seed in range(1, seeds + 1):
        fp = curve_path(inst, pop, seed)
        try:
            arr = np.load(fp)
            if len(arr) < gens + 1:
                continue
            vals = [float(arr[min(g, len(arr)-1)]) for g in checkpoints]
            rows.append(vals); used += 1
        except Exception:
            continue

    if not rows:
        return None, 0
    rows = np.array(rows)
    agg = rows.mean(axis=0) if reducer == "mean" else rows.min(axis=0)
    return agg, used

def run_grid(seeds: int, gens: int, checkpoints: List[int], reducer: str, run_missing: bool):
    
    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    SEP = "-" * 112

    # Optionally generate missing runs
    if run_missing:
        for inst in DATASETS:
            for pop in POPS:
                for seed in range(1, seeds + 1):
                    ensure_run(inst, pop, seed, gens)

    # Summarize
    hdr_cells = [f"Gen={g}" for g in checkpoints]
    header = f"{'Instance':<10} | {'Pop':>3} | {'Gens':>5} | " + " | ".join([f"{c:>10}" for c in hdr_cells])

    lines = []
    lines.append(SEP)
    lines.append("Variant: EA (Mutation-only)")
    lines.append(f"Seeds requested: {seeds} | Reducer: {reducer} | Required gens per run: {gens}")
    lines.append(SEP)
    lines.append(header)
    lines.append(SEP)

    for inst in DATASETS:
        for pop in POPS:
            agg, used = read_checkpoints(inst, pop, seeds, checkpoints, gens, reducer)
            if agg is None:
                row = f"{inst:<10} | {pop:>3} | {'NA':>5} | " + " | ".join([f"{'NA':>10}"] * len(checkpoints))
            else:
                vals = " | ".join([f"{v:>10.2f}" for v in agg])
                row = f"{inst:<10} | {pop:>3} | {gens:>5} | {vals}"
            lines.append(row)
        lines.append(SEP)

    report = "\n".join(lines)
    print(report)
    with open(OUT_TXT, "w") as f:
        f.write(report + "\n")
    print(f"Wrote {OUT_TXT}")

def parse_checkpoints(cp_str: str) -> List[int]:
    return [int(x) for x in cp_str.split(",")] if cp_str else [2000, 5000, 10000, 20000]

def main():
    ap = argparse.ArgumentParser(description="EA Variant C (mutation-only): run grid + summarize.")
    ap.add_argument("--run-grid", action="store_true", help="Run missing/short combos to completion before summarizing.")
    ap.add_argument("--seeds", type=int, default=5, help="Number of seeds per combo.")
    ap.add_argument("--gens", type=int, default=20000, help="Generations per run.")
    ap.add_argument("--checkpoints", type=str, default="2000,5000,10000,20000",
                    help="Comma-separated checkpoint gens.")
    ap.add_argument("--reducer", choices=["mean","min"], default="mean",
                    help="Aggregate across seeds using mean (default) or min.")
    args = ap.parse_args()

    checkpoints = parse_checkpoints(args.checkpoints)
    run_grid(seeds=args.seeds, gens=args.gens, checkpoints=checkpoints, reducer=args.reducer, run_missing=args.run_grid)

if __name__ == "__main__":
    main()
