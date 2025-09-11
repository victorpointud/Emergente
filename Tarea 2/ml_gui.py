import json
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from functools import partial
import matplotlib
import matplotlib.pyplot as plt
import io as _io

ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "src"
INDEX_URL = "/templates/ml_index.html"

E = 2.718281828459045
STATE_MLP = {"sizes": None, "layers": None, "last_history": None}

def sigmoid(x):
    if x >= 0:
        z = pow(E, -x)
        return 1.0 / (1.0 + z)
    else:
        z = pow(E, x)
        return z / (1.0 + z)\

def dsigmoid(y):
    return y * (1.0 - y)

def tanh(x):
    e2 = pow(E, 2 * x)
    return (e2 - 1.0) / (e2 + 1.0)

def dtanh(y):
    return 1.0 - y * y

def relu(x):
    return x if x > 0 else 0.0

def drelu(y):
    return 1.0 if y > 0 else 0.0

def act_scalar(x, kind):
    k = (kind or "").lower()
    if k in ("sigmoid", "sigmoide"):
        return sigmoid(x)
    if k == "tanh":
        return tanh(x)
    if k == "relu":
        return relu(x)
    if k in ("step", "escalon"):
        return 1.0 if x >= 0 else 0.0
    raise ValueError("Unsupported activation")

def dact_scalar(y, kind):
    k = (kind or "").lower()
    if k in ("sigmoid", "sigmoide"):
        return dsigmoid(y)
    if k == "tanh":
        return dtanh(y)
    if k == "relu":
        return drelu(y)
    if k in ("step", "escalon"):
        return 0.0
    raise ValueError("Unsupported activation")

def matvec(W, x, out_size, in_size):
    y = [0.0] * out_size
    idx = 0
    for r in range(out_size):
        s = 0.0
        for c in range(in_size):
            s += W[idx] * x[c]
            idx += 1
        y[r] = s
    return y

def add_bias(v, b):
    return [v[i] + b[i] for i in range(len(v))]

def activate_vec(v, kind):
    return [act_scalar(z, kind) for z in v]

def forward_full(x, sizes, layers, act_hidden, act_out):
    a_list = []
    z_list = []
    a = x[:]
    L = len(sizes) - 1
    for l in range(L):
        out_size = sizes[l + 1]
        in_size = sizes[l]
        W = layers[l]["W"]
        b = layers[l]["b"]
        z = add_bias(matvec(W, a, out_size, in_size), b)
        a = activate_vec(z, act_hidden if l < L - 1 else act_out)
        a_list.append(a[:])
        z_list.append(z[:])
    return a, a_list, z_list

def forward(x, sizes, layers, act_hidden, act_out):
    y, _, _ = forward_full(x, sizes, layers, act_hidden, act_out)
    return y

def accuracy(outputs, targets):
    if len(outputs[0]) == 1:
        ok = 0
        for y, t in zip(outputs, targets):
            yp = 1.0 if y[0] >= 0.5 else 0.0
            tp = 1.0 if t[0] >= 0.5 else 0.0
            if yp == tp:
                ok += 1
        return ok / len(outputs)
    else:
        ok = 0
        for y, t in zip(outputs, targets):
            ai = max(range(len(y)), key=lambda i: y[i])
            bi = max(range(len(t)), key=lambda i: t[i])
            if ai == bi:
                ok += 1
        return ok / len(outputs)

def backprop_step(x, t, sizes, layers, act_hidden, act_out, lr):
    L = len(sizes) - 1
    a0 = x[:]
    y, a_list, _ = forward_full(x, sizes, layers, act_hidden, act_out)
    deltas = [None] * L
    deltas[L - 1] = [(y[j] - t[j]) * dact_scalar(y[j], act_out) for j in range(sizes[L])]
    for l in reversed(range(L - 1)):
        out_size = sizes[l + 1]
        next_out = sizes[l + 2]
        W_next = layers[l + 1]["W"]
        delta_next = deltas[l + 1]
        d = [0.0] * out_size
        for i in range(out_size):
            s = 0.0
            for j in range(next_out):
                s += W_next[j * out_size + i] * delta_next[j]
            d[i] = s * dact_scalar(a_list[l][i], act_hidden)
        deltas[l] = d
    for l in range(L):
        out_size = sizes[l + 1]
        in_size = sizes[l]
        W = layers[l]["W"]
        b = layers[l]["b"]
        a_prev = a0 if l == 0 else a_list[l - 1]
        d = deltas[l]
        idx = 0
        for r in range(out_size):
            for c in range(in_size):
                W[idx] -= lr * d[r] * a_prev[c]
                idx += 1
        for r in range(out_size):
            b[r] -= lr * d[r]

def train_epoch(X, T, sizes, layers, act_hidden, act_out, lr):
    for x, t in zip(X, T):
        backprop_step(x, t, sizes, layers, act_hidden, act_out, lr)

def eval_dataset(X, sizes, layers, act_hidden, act_out):
    return [forward(x, sizes, layers, act_hidden, act_out) for x in X]

def parse_dataset_rows(rows):
    return [[float(v) for v in r] for r in rows]

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/mlp/plot_history.png"):
            try:
                hist = STATE_MLP.get("last_history")
                if not hist:
                    raise ValueError("No training history. Train the network first.")
                try:
                    matplotlib.use("Agg")
                except Exception:
                    self.send_response(501); self.end_headers()
                    self.wfile.write(json.dumps({"error": "Matplotlib not available"}).encode("utf-8"))
                    return
                epochs = [h["epoch"] for h in hist]
                train_acc = [h["train_acc"] if h["train_acc"] is not None else float("nan") for h in hist]
                test_acc = [h["test_acc"] if h["test_acc"] is not None else float("nan") for h in hist]
                fig = plt.figure()
                ax = fig.gca()
                ax.plot(epochs, train_acc, label="Train accuracy")
                if any(isinstance(v, float) for v in test_acc):
                    ax.plot(epochs, test_acc, label="Test accuracy")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Accuracy")
                ax.set_title("MLP accuracy per epoch")
                ax.legend()
                buf = _io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=144)
                plt.close(fig)
                png = buf.getvalue()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(png)))
                self.end_headers()
                self.wfile.write(png)
                return
            except Exception as e:
                self.send_response(400); self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
                return
        return super().do_GET()

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(n).decode("utf-8"))
        except:
            self.send_response(400); self.end_headers(); self.wfile.write(b'{"error":"Invalid JSON"}'); return

        if self.path == "/api/mlp/apply_config":
            try:
                sizes = data.get("sizes")
                layers = data.get("layers")
                if not isinstance(sizes, list) or len(sizes) < 2:
                    raise ValueError("Invalid sizes")
                if not isinstance(layers, list) or len(layers) != len(sizes) - 1:
                    raise ValueError("Invalid layers")
                for l in range(len(layers)):
                    out_size = int(sizes[l + 1]); in_size = int(sizes[l])
                    W = [float(v) for v in layers[l]["W"]]
                    b = [float(v) for v in layers[l]["b"]]
                    if len(W) != out_size * in_size:
                        raise ValueError(f"Layer {l}: invalid W shape")
                    if len(b) != out_size:
                        raise ValueError(f"Layer {l}: invalid b shape")
                    layers[l] = {"W": W, "b": b}
                STATE_MLP["sizes"] = [int(s) for s in sizes]
                STATE_MLP["layers"] = layers
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
                self.wfile.write(json.dumps({"ok": True}).encode("utf-8")); return
            except Exception as e:
                self.send_response(400); self.end_headers(); self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8")); return

        if self.path == "/api/mlp/create":
            try:
                n_in = int(data["n_in"])
                n_out = int(data["n_out"])
                n_hidden = int(data["n_hidden"])
                hidden_neurons = data["hidden_neurons"]
                if isinstance(hidden_neurons, list):
                    if len(hidden_neurons) != n_hidden:
                        raise ValueError("hidden_neurons length mismatch")
                    hs = [int(x) for x in hidden_neurons]
                else:
                    hs = [int(hidden_neurons)] * n_hidden
                sizes = [n_in] + hs + [n_out]
                layers = []
                for l in range(len(sizes) - 1):
                    out_size = sizes[l + 1]; in_size = sizes[l]
                    W = []
                    scale = 0.5
                    seed = (l + 1) * 97 + in_size * 13 + out_size * 7
                    s = seed
                    for _ in range(out_size * in_size):
                        s = (1103515245 * s + 12345) & 0x7fffffff
                        W.append(((s / 0x7fffffff) * 2 - 1) * scale)
                    b = [0.0] * out_size
                    layers.append({"W": W, "b": b})
                STATE_MLP["sizes"] = sizes
                STATE_MLP["layers"] = layers
                STATE_MLP["last_history"] = None
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
                self.wfile.write(json.dumps({"ok": True, "sizes": sizes}).encode("utf-8")); return
            except Exception as e:
                self.send_response(400); self.end_headers(); self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8")); return

        if self.path == "/api/mlp/train":
            try:
                if STATE_MLP["sizes"] is None or STATE_MLP["layers"] is None:
                    raise ValueError("Configuration not applied")
                Xtr = parse_dataset_rows(data["X_train"])
                Ytr = parse_dataset_rows(data["Y_train"])
                Xte = parse_dataset_rows(data["X_test"]) if data.get("X_test") else []
                Yte = parse_dataset_rows(data["Y_test"]) if data.get("Y_test") else []
                epochs = int(data["epochs"])
                lr = float(data.get("lr", 0.1))
                act_hidden = data.get("act_hidden", "sigmoid")
                act_out = data.get("act_out", "sigmoid")
                sizes = STATE_MLP["sizes"]
                layers = STATE_MLP["layers"]
                hist = []
                for ep in range(epochs):
                    train_epoch(Xtr, Ytr, sizes, layers, act_hidden, act_out, lr)
                    Ytr_pred = eval_dataset(Xtr, sizes, layers, act_hidden, act_out)
                    atr = accuracy(Ytr_pred, Ytr)
                    if Xte and Yte:
                        Yte_pred = eval_dataset(Xte, sizes, layers, act_hidden, act_out)
                        ate = accuracy(Yte_pred, Yte)
                    else:
                        ate = None
                    hist.append({"epoch": ep + 1, "train_acc": atr, "test_acc": ate})
                STATE_MLP["last_history"] = hist
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
                self.wfile.write(json.dumps({"history": hist}).encode("utf-8")); return
            except Exception as e:
                self.send_response(400); self.end_headers(); self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8")); return

        if self.path == "/api/mlp/eval_one":
            try:
                if STATE_MLP["sizes"] is None or STATE_MLP["layers"] is None:
                    raise ValueError("Configuration not applied")
                x = [float(v) for v in data.get("x", [])]
                act_hidden = data.get("act_hidden", "sigmoid")
                act_out = data.get("act_out", "sigmoid")
                if len(x) != STATE_MLP["sizes"][0]:
                    raise ValueError("Input size mismatch")
                y = forward(x, STATE_MLP["sizes"], STATE_MLP["layers"], act_hidden, act_out)
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
                self.wfile.write(json.dumps({"y": y}).encode("utf-8")); return
            except Exception as e:
                self.send_response(400); self.end_headers(); self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8")); return

        if self.path == "/api/mlp/eval_file":
            try:
                if STATE_MLP["sizes"] is None or STATE_MLP["layers"] is None:
                    raise ValueError("Configuration not applied")
                rows = parse_dataset_rows(data.get("rows", []))
                act_hidden = data.get("act_hidden", "sigmoid")
                act_out = data.get("act_out", "sigmoid")
                res = []
                for i, row in enumerate(rows, start=1):
                    try:
                        if len(row) != STATE_MLP["sizes"][0]:
                            raise ValueError("Input size mismatch")
                        y = forward(row, STATE_MLP["sizes"], STATE_MLP["layers"], act_hidden, act_out)
                        res.append({"idx": i, "x": row, "y": y, "err": ""})
                    except Exception as e:
                        res.append({"idx": i, "x": row, "y": None, "err": str(e)})
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
                self.wfile.write(json.dumps({"results": res}).encode("utf-8")); return
            except Exception as e:
                self.send_response(400); self.end_headers(); self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8")); return

        if self.path == "/api/mlp/save":
            try:
                if STATE_MLP["sizes"] is None or STATE_MLP["layers"] is None:
                    raise ValueError("Configuration not applied")
                obj = {"sizes": STATE_MLP["sizes"], "layers": STATE_MLP["layers"]}
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
                self.wfile.write(json.dumps(obj).encode("utf-8")); return
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