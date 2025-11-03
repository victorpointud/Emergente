import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional

import numpy as np

from .ga_functions import f1, f2, F1_BOUNDS, F2_BOUNDS

FitnessFn1D = Callable[[float], float]
FitnessFn2D = Callable[[float, float], float]

@dataclass
class GAParams:
    func: str                  
    pop_size: int = 10
    generations: int = 500
    threshold: float = 0.05    
    mr: float = 0.2            
    cr: float = 0.9             
    variability: float = 1.0    
    bounds: Tuple[float, float] = (-10.0, 10.0)
    seed: int = 42

@dataclass
class IterStats:
    iteration: int
    max_val: float
    median_val: float
    min_val: float

@dataclass
class GARunResult:
    params: GAParams
    best_vars: List[float]
    best_fitness: float
    iterations: int
    history: List[IterStats]
    final_population: List[List[float]]  
    final_fitness: List[float]           

def _clip_vec(vec: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(vec, lo, hi)

def _roulette_selection(pop: np.ndarray, fitness: np.ndarray, k: int = 2) -> List[np.ndarray]:
    fit = fitness.astype(float)
    minf = fit.min()
    if minf < 0:
        fit = fit - minf + 1e-12
    s = fit.sum()
    if s <= 0:
        idxs = np.random.choice(len(pop), size=k, replace=True)
        return [pop[i] for i in idxs]
    probs = fit / s
    idxs = np.random.choice(len(pop), size=k, replace=True, p=probs)
    return [pop[i] for i in idxs]

def _blend_crossover(p1: np.ndarray, p2: np.ndarray, cr: float = 0.9, alpha: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    if np.random.rand() > cr:
        return p1.copy(), p2.copy()
    w1 = np.random.uniform(-alpha, 1.0 + alpha, size=p1.shape)
    c1 = w1 * p1 + (1.0 - w1) * p2
    w2 = np.random.uniform(-alpha, 1.0 + alpha, size=p2.shape)
    c2 = w2 * p2 + (1.0 - w2) * p1
    return c1, c2

def _mutate(ind: np.ndarray, mr: float = 0.2, mut_range: float = 1.0) -> np.ndarray:
    if mut_range <= 0.0 or mr <= 0.0:
        return ind
    mask = np.random.rand(*ind.shape) < mr
    noise = np.random.uniform(-mut_range, mut_range, size=ind.shape)
    out = ind.copy()
    out[mask] = out[mask] + noise[mask]
    return out

def _eval_population(pop: np.ndarray, func_name: str) -> np.ndarray:
    vals = []
    if func_name == "f1":
        for v in pop:
            vals.append(f1(float(v[0])))
    else:
        for v in pop:
            vals.append(f2(float(v[0]), float(v[1])))
    return np.array(vals, dtype=float)

def run_ga(params: GAParams) -> GARunResult:

    rng = np.random.default_rng(params.seed)
    random.seed(params.seed)
    np.random.seed(params.seed)

    lo, hi = float(params.bounds[0]), float(params.bounds[1])
    dims = 1 if params.func == "f1" else 2

    pop = rng.uniform(lo, hi, size=(params.pop_size, dims))
    history: List[IterStats] = []
    best_prev: Optional[float] = None

    for it in range(1, params.generations + 1):
        fit = _eval_population(pop, params.func)
        order = np.argsort(-fit)
        pop = pop[order]
        fit = fit[order]

        mx = float(fit.max())
        md = float(np.median(fit))
        mn = float(fit.min())
        history.append(IterStats(iteration=it, max_val=mx, median_val=md, min_val=mn))

        if best_prev is not None and abs(mx - best_prev) < params.threshold:
            break
        best_prev = mx

        new_pop: List[np.ndarray] = [pop[0].copy()]
        if params.pop_size > 1:
            new_pop.append(pop[1].copy())

        while len(new_pop) < params.pop_size:
            p1, p2 = _roulette_selection(pop, fit, k=2)
            c1, c2 = _blend_crossover(p1, p2, cr=params.cr, alpha=0.3)
            c1 = _mutate(c1, mr=params.mr, mut_range=params.variability)
            c2 = _mutate(c2, mr=params.mr, mut_range=params.variability)
            c1 = _clip_vec(c1, lo, hi)
            c2 = _clip_vec(c2, lo, hi)
            new_pop.append(c1)
            if len(new_pop) < params.pop_size:
                new_pop.append(c2)
        pop = np.array(new_pop, dtype=float)

    fit = _eval_population(pop, params.func)
    order = np.argsort(-fit)
    pop = pop[order]
    fit = fit[order]

    best_idx = int(np.argmax(fit))
    best = pop[best_idx].tolist()
    best_fit = float(fit[best_idx])

    return GARunResult(
        params=params,
        best_vars=best,
        best_fitness=best_fit,
        iterations=len(history),
        history=history,
        final_population=[p.tolist() for p in pop],
        final_fitness=fit.tolist(),
    )

def save_population_json(path: Path, result: GARunResult) -> None:
    data = {
        "func": result.params.func,
        "params": asdict(result.params),
        "iterations": result.iterations,
        "best_vars": result.best_vars,
        "best_fitness": result.best_fitness,
        "final_population": result.final_population,
        "final_fitness": result.final_fitness,
        "history": [asdict(h) for h in result.history],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def load_population_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))