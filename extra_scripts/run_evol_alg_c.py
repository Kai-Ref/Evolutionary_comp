# Run EA Variant C (mutation-only) grid in parallel — no shell script needed.

import os, argparse, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

# Keep NumPy/BLAS single-threaded per process to avoid oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from src.EvolutionaryAlgorithmC import EvolutionaryAlgorithmC as EAC  # your class


def curve_path(logdir: str, inst: str, pop: int, seed: int) -> str:
    return os.path.join(logdir, inst, f"pop_{pop}", f"seed_{seed}", "best_cost_per_generation.npy")


def run_one(inst: str, pop: int, seed: int, gens: int, logdir: str, precompute: bool) -> str:
    """
    Run EAC for a single (instance, pop, seed) if missing/short.
    Returns the path to best_cost_per_generation.npy.
    """
    out = curve_path(logdir, inst, pop, seed)
    try:
        arr = np.load(out)
        if len(arr) >= gens + 1:
            return out  # already done
    except Exception:
        pass

    ds_path = os.path.join("datasets", f"{inst}.tsp")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ea = EAC(filepath=ds_path,
             population_size=pop,
             precompute_distances=precompute,
             seed=seed,
             log_dir=logdir)
    ea.solve(max_generations=gens)
    return out


def main():
    parser = argparse.ArgumentParser(description="Parallel EA-C grid runner (Python-only).")
    parser.add_argument("--instances", nargs="*", default=[
        "eil51","eil76","eil101","st70",
        "kroA100","kroC100","kroD100","lin105",
        "pcb442","pr2392","usa13509"
    ], help="TSPlib instance names (without .tsp).")
    parser.add_argument("--pops", nargs="*", type=int, default=[20,50,100,200],
                        help="Population sizes.")
    parser.add_argument("--seeds", type=int, default=30, help="Seeds per combo.")
    parser.add_argument("--gens", type=int, default=20000, help="Generations per run.")
    parser.add_argument("--logdir", type=str, default="results/ea_variant_c",
                        help="Output root for curves.")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                        help="Parallel workers (processes).")
    parser.add_argument("--no-precompute", action="store_true",
                        help="Disable distance matrix precomputation.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip starting work for combos that already have >= gens+1 points.")
    parser.add_argument("--evaluate", action="store_true",
                        help="Call evaluation.py afterwards to print the table.")
    parser.add_argument("--eval-script", type=str, default="evaluation.py",
                        help="Path to your evaluation.py (the schedule + Min/Mean + checkpoints one).")
    parser.add_argument("--eval-out", type=str, default="results/EA_mutation_results.txt",
                        help="Where to save the evaluation table.")
    parser.add_argument("--checkpoints", type=str, default="2000,5000,10000,20000",
                        help="Comma-separated checkpoints for evaluation.")
    parser.add_argument("--reducer", choices=["mean","min"], default="mean",
                        help="Aggregate across seeds for checkpoints.")
    args = parser.parse_args()

    os.environ.setdefault("PYTHONPATH", os.getcwd())
    os.makedirs(args.logdir, exist_ok=True)

    # Build job list
    jobs = []
    for inst in args.instances:
        for pop in args.pops:
            for seed in range(1, args.seeds + 1):
                if args.skip_existing:
                    out = curve_path(args.logdir, inst, pop, seed)
                    try:
                        arr = np.load(out)
                        if len(arr) >= args.gens + 1:
                            continue
                    except Exception:
                        pass
                jobs.append((inst, pop, seed))

    print(f"[EA-C] total jobs to run: {len(jobs)} | workers: {args.jobs}")
    if not jobs:
        print("Nothing to do — all combos already complete.")

    # Run in parallel
    precompute = not args.no_precompute
    errors = []
    if jobs:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {
                ex.submit(run_one, inst, pop, seed, args.gens, args.logdir, precompute): (inst, pop, seed)
                for (inst, pop, seed) in jobs
            }
            for fut in as_completed(futs):
                inst, pop, seed = futs[fut]
                try:
                    path = fut.result()
                    print(f"✓ {inst} | pop={pop} | seed={seed} → {path}")
                except Exception:
                    tb = traceback.format_exc()
                    print(f"✗ {inst} | pop={pop} | seed={seed}\n{tb}")
                    errors.append((inst, pop, seed))

    if errors:
        print(f"[WARN] {len(errors)} jobs failed. They can be retried with --skip-existing to skip finished ones.")

    # Optional evaluation at the end (same format as your evaluation.py table)
    if args.evaluate:
        import subprocess, sys
        os.makedirs(os.path.dirname(args.eval_out), exist_ok=True)
        cmd = [
            sys.executable, args.eval_script,
            "--variant_dir", args.logdir,
            "--out", args.eval_out,
            "--seeds", str(args.seeds),
            "--checkpoints", args.checkpoints,
            "--reducer", args.reducer,
        ]
        print(f"[Eval] Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print("[Eval] evaluation.py failed:", e)


if __name__ == "__main__":
    main()
