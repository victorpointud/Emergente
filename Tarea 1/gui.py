import json
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from functools import partial

ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "src"
INDEX_URL = "/templates/index.html"

STATE = {"bias": None, "weights": None}

def weighted_sum(x, w, b):
    if len(x) != len(w):
        raise ValueError(f"Incompatible dimensions: len(x)={len(x)} vs len(w)={len(w)}")
    s = b
    for i in range(len(x)):
        s += x[i] * w[i]
    return s

def activate(v, kind):
    t = (kind or "").lower()
    if t in ("step", "escalon"):
        return 1.0 if v >= 0 else 0.0
    if t in ("sigmoid", "sigmoide"):
        if v >= 0:
            z = pow(2.718281828459045, -v)
            return 1.0 / (1.0 + z)
        z = pow(2.718281828459045, v)
        return z / (1.0 + z)
    raise ValueError("Unsupported activation. Use 'step' or 'sigmoid'.")

class Handler(SimpleHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            data = json.loads(body)
        except Exception:
            self.send_response(400); self.end_headers(); self.wfile.write(b'{"error":"Invalid JSON"}'); return

        if self.path == "/api/apply_config":
            try:
                bias = float(data.get("bias", None))
                weights = [float(v) for v in data.get("weights", [])]
                if not weights or bias is None:
                    raise ValueError("Missing bias or weights")
                STATE["bias"] = bias
                STATE["weights"] = weights
                resp = {"ok": True, "n": len(weights)}
                self.send_response(200); self.send_header("Content-Type","application/json"); self.end_headers()
                self.wfile.write(json.dumps(resp).encode("utf-8"))
                return
            except Exception as e:
                self.send_response(400); self.end_headers(); self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8")); return

        if self.path == "/api/eval_one":
            try:
                if STATE["bias"] is None or STATE["weights"] is None:
                    raise ValueError("Configuration not applied")
                x = [float(v) for v in data.get("x", [])]
                act = data.get("activation", "step")
                z = weighted_sum(x, STATE["weights"], STATE["bias"])
                y = activate(z, act)
                resp = {"z": z, "y": y}
                self.send_response(200); self.send_header("Content-Type","application/json"); self.end_headers()
                self.wfile.write(json.dumps(resp).encode("utf-8"))
                return
            except Exception as e:
                self.send_response(400); self.end_headers(); self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8")); return

        if self.path == "/api/eval_file":
            try:
                if STATE["bias"] is None or STATE["weights"] is None:
                    raise ValueError("Configuration not applied")
                rows = data.get("rows", [])
                act = data.get("activation", "step")
                results = []
                for i, row in enumerate(rows, start=1):
                    try:
                        x = [float(v) for v in row]
                        z = weighted_sum(x, STATE["weights"], STATE["bias"])
                        y = activate(z, act)
                        results.append({"idx": i, "x": x, "z": z, "y": y, "err": ""})
                    except Exception as e:
                        results.append({"idx": i, "x": row, "z": None, "y": None, "err": str(e)})
                self.send_response(200); self.send_header("Content-Type","application/json"); self.end_headers()
                self.wfile.write(json.dumps({"results": results}).encode("utf-8"))
                return
            except Exception as e:
                self.send_response(400); self.end_headers(); self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8")); return

        self.send_response(404); self.end_headers(); self.wfile.write(b'{"error":"Not Found"}')

def main(port=8000):
    h = partial(Handler, directory=str(WEB_ROOT))
    httpd = ThreadingHTTPServer(("127.0.0.1", port), h)
    print(f"Serving {WEB_ROOT} at http://127.0.0.1:{port}{INDEX_URL}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()