import io
import json
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from functools import partial
from pathlib import Path
import math
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .ga_core import GAParams, run_ga, GARunResult
from .ga_functions import f1, f2, F1_BOUNDS, F2_BOUNDS

ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT.parent             
INDEX_URL = "/template/ga.html"


STATE = {
    "last_result": None,     
}

def _fig_png():
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def _plot_history_png():
    res: GARunResult | None = STATE["last_result"]
    if not res or not res.history:
        return None
    xs  = [h.iteration for h in res.history]
    mxs = [h.max_val for h in res.history]
    mds = [h.median_val for h in res.history]
    mns = [h.min_val for h in res.history]
    plt.clf()
    plt.plot(xs, mxs, label="max")
    plt.plot(xs, mds, label="median")
    plt.plot(xs, mns, label="min")
    plt.xlabel("iteration")
    plt.ylabel("fitness")
    plt.title("GA History")
    plt.legend()
    return _fig_png()

def _plot_surface_png():
    res: GARunResult | None = STATE["last_result"]
    if not res:
        return None

    func = res.params.func
    lo, hi = res.params.bounds
    best = res.best_vars if res.best_vars else None

    plt.clf()
    if func == "f1":
        xs = np.linspace(lo, hi, 800)
        ys = [(math.sin(x) ** 2) / x if abs(x) > 1e-12 else 0.0 for x in xs]
        plt.plot(xs, ys, label="f1(x) = sin(x)^2 / x")
        if best is not None:
            bx = float(best[0])
            by = (math.sin(bx) ** 2) / bx if abs(bx) > 1e-12 else 0.0
            plt.scatter([bx], [by], c="tab:red", s=50, edgecolors="k",
                        label=f"best ({bx:.4f}, {by:.4f})")
        plt.xlabel("x")
        plt.ylabel("f1(x)")
        plt.title("f1 curve")
        plt.legend()
    else:
        N = 180
        X = np.linspace(lo, hi, N)
        Y = np.linspace(lo, hi, N)
        XX, YY = np.meshgrid(X, Y)
        ZZ = 20.0 + XX - 10.0*np.cos(2.0*math.pi*XX) + YY - 10.0*np.cos(2.0*math.pi*YY)
        cs = plt.contourf(XX, YY, ZZ, levels=24, cmap="viridis")
        plt.colorbar(cs, label="f2(x,y)")
        if best is not None and len(best) == 2:
            plt.scatter([best[0]], [best[1]], c="tab:red", s=55, edgecolors="k",
                        label=f"best ({best[0]:.3f}, {best[1]:.3f})")
            plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("f2 contour")
    return _fig_png()

class Handler(SimpleHTTPRequestHandler):

    def do_GET(self):
        p = self.path.split("?")[0]

        if p == "/":
            self.send_response(302)
            self.send_header("Location", INDEX_URL)
            self.end_headers()
            return

        if p == "/api/ga/plot_history.png":
            png = _plot_history_png()
            if png is None:
                self.send_response(404); self.end_headers(); return
            self._png(png); return

        if p == "/api/ga/plot_surface.png":
            png = _plot_surface_png()
            if png is None:
                self.send_response(404); self.end_headers(); return
            self._png(png); return

        return super().do_GET()

    def _png(self, png_bytes: bytes):
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.end_headers()
        self.wfile.write(png_bytes)

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(n).decode("utf-8")
            data = json.loads(raw) if raw else {}
        except Exception:
            self.send_response(400); self.end_headers()
            self.wfile.write(b'{"error":"Invalid JSON"}')
            return

        try:
            if self.path == "/api/ga/run":
                func = str(data.get("func", "f1"))
                pop = int(data.get("pop", 10))
                gens = int(data.get("iters", 500))
                thr = float(data.get("threshold", 0.05))
                mr = float(data.get("mr", 0.2))
                cr = float(data.get("cr", 0.9))
                variability = float(data.get("mutvar", 1.0))
                bounds = data.get("bounds")
                if not (isinstance(bounds, (list, tuple)) and len(bounds) == 2):
                    bounds = (-10.0, 10.0)

                params = GAParams(
                    func=("f1" if func == "f1" else "f2"),
                    pop_size=pop,
                    generations=gens,
                    threshold=thr,
                    mr=mr,
                    cr=cr,
                    variability=variability,
                    bounds=(float(bounds[0]), float(bounds[1])),
                    seed=42,
                )
                result: GARunResult = run_ga(params)
                STATE["last_result"] = result

                out = {
                    "history": [{"iter": h.iteration,
                                 "max": h.max_val,
                                 "median": h.median_val,
                                 "min": h.min_val} for h in result.history],
                    "best_vars": result.best_vars,
                    "best_fitness": result.best_fitness,
                    "iterations": result.iterations,
                }
                self._ok(out); return

            if self.path == "/api/ga/save":
                payload = data or {}
                filename = str(payload.get("filename", "ga_last.json"))
                res: GARunResult | None = STATE["last_result"]
                if not res:
                    raise ValueError("No GA result to save. Run first.")
                obj = {
                    "params": {
                        "func": res.params.func,
                        "pop_size": res.params.pop_size,
                        "generations": res.params.generations,
                        "threshold": res.params.threshold,
                        "mr": res.params.mr,
                        "cr": res.params.cr,
                        "variability": res.params.variability,
                        "bounds": list(res.params.bounds),
                        "seed": res.params.seed,
                    },
                    "best_vars": res.best_vars,
                    "best_fitness": res.best_fitness,
                    "iterations": res.iterations,
                    "history": [{"iter": h.iteration,
                                 "max": h.max_val,
                                 "median": h.median_val,
                                 "min": h.min_val} for h in res.history],
                    "final_population": res.final_population,
                    "final_fitness": res.final_fitness,
                }
                Path(filename).write_text(json.dumps(obj, indent=2), encoding="utf-8")
                self._ok({"saved": filename}); return

            self.send_response(404); self.end_headers()
            self.wfile.write(b'{"error":"Not Found"}')
        except Exception as e:
            self.send_response(400); self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

    def _ok(self, payload: dict):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

def main(port: int = 9200):
    h = partial(Handler, directory=str(WEB_ROOT))
    httpd = ThreadingHTTPServer(("127.0.0.1", port), h)
    print(f"Serving at http://127.0.0.1:{port}{INDEX_URL}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()