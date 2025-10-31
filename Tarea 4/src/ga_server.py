import io
import json
import math
import random
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from functools import partial
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ga_functions import f1, f2

ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT.parent  # serve whole project root (so ../template and ../static are accessible)
INDEX_URL = "/template/ga.html"

STATE = {
    "last_history": [],     # list of dicts {iter, max, median, min}
    "last_func": "f1",      # f1 or f2
    "last_best": None,      # (vars array)
    "bounds": (-10.0, 10.0)
}

# -----------------------------
# GA core (maximization)
# -----------------------------
def _clip_vec(vec, lo, hi):
    return np.clip(vec, lo, hi)

def _roulette_selection(pop, fitness, k=2):
    fit = np.array(fitness, dtype=float)
    # shift if negative
    minf = fit.min()
    if minf < 0:
        fit = fit - minf + 1e-12
    s = fit.sum()
    if s <= 0:
        # fallback: uniform if degenerate
        idxs = np.random.choice(len(pop), size=k, replace=True)
        return [pop[i] for i in idxs]
    probs = fit / s
    idxs = np.random.choice(len(pop), size=k, replace=True, p=probs)
    return [pop[i] for i in idxs]

def _blend_crossover(p1, p2, cr=0.9, alpha=0.5):
    if np.random.rand() > cr:
        return p1.copy(), p2.copy()
    # BLX-α style (simplified): convex combinations
    w = np.random.uniform(0.0 - alpha, 1.0 + alpha, size=p1.shape)
    c1 = w * p1 + (1.0 - w) * p2
    w2 = np.random.uniform(0.0 - alpha, 1.0 + alpha, size=p1.shape)
    c2 = w2 * p2 + (1.0 - w2) * p1
    return c1, c2

def _mutate(ind, mr=0.2, mut_range=0.5):
    if mut_range <= 0: 
        return ind
    mask = np.random.rand(*ind.shape) < mr
    noise = np.random.uniform(-mut_range, mut_range, size=ind.shape)
    out = ind.copy()
    out[mask] = out[mask] + noise[mask]
    return out

def _eval_population(pop, func_name):
    vals = []
    if func_name == "f1":
        for v in pop:
            vals.append(f1(float(v[0])))
    else:
        for v in pop:
            vals.append(f2(float(v[0]), float(v[1])))
    return np.array(vals, dtype=float)

def run_ga(func_name="f1", pop_size=30, iters=300, threshold=1e-3, mr=0.2, cr=0.9, mutvar=0.5, bounds=(-10,10), seed=42):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)

    lo, hi = float(bounds[0]), float(bounds[1])
    dims = 1 if func_name == "f1" else 2

    pop = rng.uniform(lo, hi, size=(pop_size, dims))
    history = []
    best_fit_prev = None

    for it in range(1, iters+1):
        fit = _eval_population(pop, func_name)
        order = np.argsort(-fit)  # descending
        pop = pop[order]
        fit = fit[order]

        mx = float(fit.max())
        md = float(np.median(fit))
        mn = float(fit.min())
        history.append({"iter": it, "max": mx, "median": md, "min": mn})

        # convergence check
        if best_fit_prev is not None and abs(mx - best_fit_prev) < threshold:
            break
        best_fit_prev = mx

        # next generation
        new_pop = []
        # elitism: keep top 2
        new_pop.append(pop[0].copy())
        if pop_size > 1:
            new_pop.append(pop[1].copy())

        while len(new_pop) < pop_size:
            p1, p2 = _roulette_selection(pop, fit, k=2)
            c1, c2 = _blend_crossover(p1, p2, cr=cr, alpha=0.3)
            c1 = _mutate(c1, mr=mr, mut_range=mutvar)
            c2 = _mutate(c2, mr=mr, mut_range=mutvar)
            c1 = _clip_vec(c1, lo, hi)
            c2 = _clip_vec(c2, lo, hi)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)
        pop = np.array(new_pop, dtype=float)

    # final evaluation
    fit = _eval_population(pop, func_name)
    best_idx = int(np.argmax(fit))
    best = pop[best_idx].tolist()
    best_fit = float(fit[best_idx])

    STATE["last_history"] = history
    STATE["last_func"] = func_name
    STATE["last_best"] = best
    STATE["bounds"] = (lo, hi)

    return {
        "history": history,
        "best_vars": best,
        "best_fitness": best_fit,
        "iterations": len(history)
    }

# -----------------------------
# Plot helpers
# -----------------------------
def _fig_png():
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def plot_history_png():
    hist = STATE["last_history"]
    if not hist:
        return None
    xs  = [h["iter"] for h in hist]
    mxs = [h["max"] for h in hist]
    mds = [h["median"] for h in hist]
    mns = [h["min"] for h in hist]

    plt.clf()
    plt.plot(xs, mxs, label="max")
    plt.plot(xs, mds, label="median")
    plt.plot(xs, mns, label="min")
    plt.xlabel("iteration"); plt.ylabel("fitness"); plt.title("GA History")
    plt.legend()
    return _fig_png()

def plot_surface_png():
    func = STATE["last_func"]
    lo, hi = STATE["bounds"]
    best = STATE["last_best"]

    plt.clf()
    if func == "f1":
        xs = np.linspace(lo, hi, 600)
        ys = []
        for x in xs:
            ys.append(f1(float(x)))
        plt.plot(xs, ys, label="f1(x)")
        if best is not None:
            bx = best[0]
            by = f1(float(bx))
            plt.scatter([bx],[by], c="tab:red", label=f"best ({bx:.3f}, {by:.3f})")
        plt.xlabel("x"); plt.ylabel("f1(x)")
        plt.title("f1 curve")
        plt.legend()
    else:
        # contour for f2
        N = 160
        X = np.linspace(lo, hi, N)
        Y = np.linspace(lo, hi, N)
        XX, YY = np.meshgrid(X, Y)
        ZZ = 20.0 + XX - 10.0*np.cos(2.0*math.pi*XX) + YY - 10.0*np.cos(2.0*math.pi*YY)
        cs = plt.contourf(XX, YY, ZZ, levels=20, cmap="viridis")
        plt.colorbar(cs, label="f2(x,y)")
        if best is not None:
            plt.scatter([best[0]],[best[1]], c="tab:red", s=50, edgecolors="k", label=f"best ({best[0]:.3f}, {best[1]:.3f})")
            plt.legend()
        plt.xlabel("x"); plt.ylabel("y"); plt.title("f2 contour")
    return _fig_png()

# -----------------------------
# HTTP Handler
# -----------------------------
class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/":
            self.send_response(302)
            self.send_header("Location", INDEX_URL)
            self.end_headers()
            return

        if p == "/api/ga/plot_history.png":
            png = plot_history_png()
            if png is None:
                self.send_response(404); self.end_headers(); return
            self._png(png); return

        if p == "/api/ga/plot_surface.png":
            png = plot_surface_png()
            if png is None:
                self.send_response(404); self.end_headers(); return
            self._png(png); return

        return super().do_GET()

    def _png(self, png_bytes):
        self.send_response(200)
        self.send_header("Content-Type","image/png")
        self.end_headers()
        self.wfile.write(png_bytes)

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length","0"))
            raw = self.rfile.read(n).decode("utf-8")
            data = json.loads(raw) if raw else {}
        except Exception:
            self.send_response(400); self.end_headers(); self.wfile.write(b'{"error":"Invalid JSON"}'); return

        try:
            if self.path == "/api/ga/run":
                func = data.get("func","f1")
                pop  = int(data.get("pop",30))
                iters = int(data.get("iters",300))
                thr = float(data.get("threshold",1e-3))
                mr = float(data.get("mr",0.2))
                cr = float(data.get("cr",0.9))
                mutvar = float(data.get("mutvar",0.5))
                bounds = data.get("bounds", [-10,10])
                if func not in ("f1","f2"):
                    raise ValueError("func must be f1 or f2")
                if not (isinstance(bounds, (list,tuple)) and len(bounds)==2):
                    bounds = [-10,10]
                res = run_ga(func_name=func, pop_size=pop, iters=iters, threshold=thr,
                             mr=mr, cr=cr, mutvar=mutvar, bounds=(float(bounds[0]), float(bounds[1])))
                self._ok(res); return

            self.send_response(404); self.end_headers(); self.wfile.write(b'{"error":"Not Found"}')
        except Exception as e:
            self.send_response(400); self.end_headers(); self.wfile.write(json.dumps({"error":str(e)}).encode("utf-8"))

    def _ok(self, payload):
        self.send_response(200)
        self.send_header("Content-Type","application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

def main(port=9100):
    h = partial(Handler, directory=str(WEB_ROOT))
    httpd = ThreadingHTTPServer(("127.0.0.1", port), h)
    print(f"Serving at http://127.0.0.1:{port}{INDEX_URL}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main() 