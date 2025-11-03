import argparse
from pathlib import Path
from typing import Tuple

from .ga_core import GAParams, run_ga, save_population_json
from .ga_functions import F1_BOUNDS, F2_BOUNDS


def _bounds_for(func: str) -> Tuple[float, float]:
    if func == "f1":
        return F1_BOUNDS
    return F2_BOUNDS

def _print_iter_line(it, mx, md, mn):
    print(f"Iter {it:4d} | max={mx: .6f} | median={md: .6f} | min={mn: .6f}")

def _print_summary(result):
    p = result.params
    print("\n=== FINAL SUMMARY ===")
    print(f"Function        : {p.func}")
    print(f"Population size : {p.pop_size}")
    print(f"Generations     : {p.generations}")
    print(f"Threshold       : {p.threshold}")
    print(f"Mutation rate   : {p.mr}")
    print(f"Crossover rate  : {p.cr}")
    print(f"Variability     : {p.variability}")
    print(f"Bounds          : {p.bounds}")
    print(f"Seed            : {p.seed}")
    print(f"Iterations ran  : {result.iterations}")
    print(f"Best vars       : {result.best_vars}")
    print(f"Best fitness    : {result.best_fitness:.6f}")

    print("\n--- Final population (sorted best → worst) ---")
    for i, (ind, fit) in enumerate(zip(result.final_population, result.final_fitness), start=1):
        vars_txt = ", ".join(f"{v:.6f}" for v in ind)
        print(f"{i:02d}. x = [{vars_txt}] | f = {fit:.6f}")
    print("------------------------------------------------\n")


def run_case(func: str, variability: float, out_path: Path, args) -> None:
    params = GAParams(
        func=func,
        pop_size=args.pop,
        generations=args.generations,
        threshold=args.threshold,
        mr=args.mr,
        cr=args.cr,
        variability=variability,
        bounds=_bounds_for(func),
        seed=args.seed
    )
    result = run_ga(params)

    for h in result.history:
        _print_iter_line(h.iteration, h.max_val, h.median_val, h.min_val)

    _print_summary(result)
    save_population_json(out_path, result)
    print(f"Saved: {out_path.resolve()}")

def run_all_four(args) -> None:
    low = args.low if args.low is not None else 0.5
    high = args.high if args.high is not None else 1.5

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Running f1 with LOW variability ===")
    run_case("f1", low, out_dir / "f1_low.json", args)

    print("\n=== Running f1 with HIGH variability ===")
    run_case("f1", high, out_dir / "f1_high.json", args)

    print("\n=== Running f2 with LOW variability ===")
    run_case("f2", low, out_dir / "f2_low.json", args)

    print("\n=== Running f2 with HIGH variability ===")
    run_case("f2", high, out_dir / "f2_high.json", args)


def build_parser():
    ap = argparse.ArgumentParser(
        description="Genetic Algorithm"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_all = sub.add_parser("run-all", help="Run all four cases (f1/f2 × low/high variability)")
    p_all.add_argument("--pop", type=int, default=10, help="Population size (default: 10)")
    p_all.add_argument("--generations", type=int, default=500, help="Max generations (default: 500)")
    p_all.add_argument("--threshold", type=float, default=0.05, help="Convergence threshold (default: 0.05)")
    p_all.add_argument("--mr", type=float, default=0.2, help="Mutation rate (default: 0.2)")
    p_all.add_argument("--cr", type=float, default=0.9, help="Crossover rate (default: 0.9)")
    p_all.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    p_all.add_argument("--low", type=float, help="LOW variability value (<1). Default: 0.5")
    p_all.add_argument("--high", type=float, help="HIGH variability value (>1). Default: 1.5")
    p_all.add_argument("--outdir", type=str, default=".", help="Output directory (default: current)")

    p_one = sub.add_parser("run-one", help="Run a single case and save JSON")
    p_one.add_argument("--func", choices=["f1", "f2"], required=True, help="Objective function")
    p_one.add_argument("--variability", type=float, required=True, help="Mutation variability (e.g., 0.5 or 1.5)")
    p_one.add_argument("--out", type=str, required=True, help="Output JSON path")

    p_one.add_argument("--pop", type=int, default=10)
    p_one.add_argument("--generations", type=int, default=500)
    p_one.add_argument("--threshold", type=float, default=0.05)
    p_one.add_argument("--mr", type=float, default=0.2)
    p_one.add_argument("--cr", type=float, default=0.9)
    p_one.add_argument("--seed", type=int, default=42)

    return ap

def main():
    ap = build_parser()
    args = ap.parse_args()

    if args.cmd == "run-all":
        run_all_four(args)
    elif args.cmd == "run-one":
        out_path = Path(args.out)
        run_case(args.func, args.variability, out_path, args)

if __name__ == "__main__":
    main()