# src/mlp_server.py
import io, json, math
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from functools import partial
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT        
INDEX_URL = "/src/templates/mlp.html"

STATE = {
    "sizes": None,     
    "W": None,         
    "b": None,          
    "act_hidden": "relu",
    "act_out": None,    
    "mu": None,         
    "std": None,        
    "history": [],      
}

def zscore_fit(X):
    mu = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-12
    return mu, std

def zscore_apply(X, mu, std):
    return (X - mu) / (std + 1e-12)

def activation(a, kind):
    if kind == "relu":
        return np.maximum(0.0, a)
    if kind == "sigmoid":
        out = np.empty_like(a)
        pos = a >= 0
        out[pos]  = 1.0 / (1.0 + np.exp(-a[pos]))
        ez = np.exp(a[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out
    if kind == "tanh":
        return np.tanh(a)
    if kind == "linear":
        return a
    raise ValueError("Unknown activation")

def dactivation(a, kind):
    if kind == "relu":
        return (a > 0).astype(a.dtype)
    if kind == "sigmoid":
        return a * (1.0 - a)
    if kind == "tanh":
        return 1.0 - np.power(a, 2.0)
    if kind == "linear":
        return np.ones_like(a)
    raise ValueError("Unknown dactivation")

def softmax(z):
    z2 = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z2)
    return ez / (np.sum(ez, axis=1, keepdims=True) + 1e-12)

def forward(X):
    A = [X]
    Z = []
    for l in range(len(STATE["W"]) - 1):
        z = A[-1] @ STATE["W"][l] + STATE["b"][l]
        a = activation(z, STATE["act_hidden"])
        Z.append(z); A.append(a)
    zL = A[-1] @ STATE["W"][-1] + STATE["b"][-1]
    if STATE["act_out"] == "sigmoid":
        aL = activation(zL, "sigmoid")
    else:
        aL = softmax(zL)
    Z.append(zL); A.append(aL)
    return Z, A

def predict_classes(X):
    _, A = forward(X)
    out = A[-1]
    if STATE["act_out"] == "sigmoid":
        return (out >= 0.5).astype(int).reshape(-1)
    return np.argmax(out, axis=1)

def bce_sigmoid_grad(A_last, Y):

    eps = 1e-12
    yhat = np.clip(A_last, eps, 1.0 - eps)
    loss = -np.mean(Y * np.log(yhat) + (1 - Y) * np.log(1 - yhat))
    dZ = (A_last - Y)  
    return loss, dZ

def ce_softmax_grad(A_last, Y_onehot):
    eps = 1e-12
    yhat = np.clip(A_last, eps, 1.0 - eps)
    loss = -np.mean(np.sum(Y_onehot * np.log(yhat), axis=1))
    dZ = (A_last - Y_onehot) 
    return loss, dZ

def one_hot(y, C):
    Y = np.zeros((y.size, C), dtype=np.float64)
    Y[np.arange(y.size), y.astype(int)] = 1.0
    return Y

def accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())

def ensure_model(sizes):
    if STATE["sizes"] == sizes and STATE["W"] and STATE["b"]:
        return
    STATE["sizes"] = sizes
    STATE["W"] = []
    STATE["b"] = []
    rng = np.random.default_rng(42)
    for l in range(len(sizes)-1):
        fan_in, fan_out = sizes[l], sizes[l+1]
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        STATE["W"].append(rng.uniform(-limit, limit, size=(fan_in, fan_out)))
        STATE["b"].append(np.zeros((1, fan_out), dtype=np.float64))
    STATE["history"] = []

def apply_config_json(payload):
    sizes = payload.get("sizes")
    layers = payload.get("layers", [])
    if not sizes or not isinstance(sizes, list) or len(sizes) < 2:
        raise ValueError("Invalid 'sizes'.")
    ensure_model(sizes)
    if layers:
        if len(layers) != len(sizes) - 1:
            raise ValueError("layers length mismatch.")
        W, B = [], []
        for li, lay in enumerate(layers):
            Wflat = np.array(lay["W"], dtype=np.float64)
            b = np.array(lay["b"], dtype=np.float64).reshape(1, sizes[li+1])
            W = Wflat.reshape(sizes[li], sizes[li+1])
            W, b = W, b
            STATE["W"][li] = W
            STATE["b"][li] = b

def save_config_json():
    sizes = STATE["sizes"]
    layers = []
    for l in range(len(STATE["W"])):
        W = STATE["W"][l].reshape(-1).tolist()
        b = STATE["b"][l].reshape(-1).tolist()
        layers.append({"W": W, "b": b})
    return {"sizes": sizes, "layers": layers}

def split_xy_if_needed(rows):
    if not rows or not isinstance(rows[0], list):
        raise ValueError("Invalid CSV rows.")
    X = [r[:-1] for r in rows]
    Y = [[r[-1]] for r in rows]
    return X, Y

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/mlp/plot_history.png"):
            if not STATE["history"]:
                self.send_response(404); self.end_headers(); return
            buf = io.BytesIO()
            plt.clf()
            xs = [h["epoch"] for h in STATE["history"]]
            tr = [h["train_acc"] for h in STATE["history"]]
            te = [h["test_acc"] for h in STATE["history"] if h["test_acc"] is not None]
            plt.plot(xs, tr, label="train acc")
            if te: plt.plot(xs[:len(te)], te, label="test acc")
            plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.title("Training history"); plt.legend()
            plt.tight_layout()
            plt.savefig(buf, format="png"); buf.seek(0)
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.end_headers()
            self.wfile.write(buf.read())
            return
        return super().do_GET()

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length","0"))
            body = self.rfile.read(n).decode("utf-8")
            data = json.loads(body) if body else {}
        except Exception:
            self.send_response(400); self.end_headers(); self.wfile.write(b'{"error":"Invalid JSON"}'); return

        try:
            if self.path == "/api/mlp/create":
                n_in = int(data["n_in"]); n_out = int(data["n_out"])
                n_hidden = int(data["n_hidden"]); hs = data.get("hidden_neurons", [])
                if len(hs) != n_hidden: 
                    raise ValueError("hidden_neurons length must match n_hidden")
                sizes = [n_in] + [int(max(1,h)) for h in hs] + [n_out]
                ensure_model(sizes)
                STATE["history"] = []
                self._ok({"sizes": sizes}); return

            if self.path == "/api/mlp/apply_config":
                apply_config_json(data)
                STATE["history"] = []
                self._ok({"ok": True}); return

            if self.path == "/api/mlp/train":
                Xtr = data.get("X_train", [])
                Ytr = data.get("Y_train", [])
                Xte = data.get("X_test", [])
                Yte = data.get("Y_test", [])
                epochs = int(data.get("epochs", 50))
                lr = float(data.get("lr", 0.05))
                act_hidden = data.get("act_hidden", "relu").lower()
                act_out_req = data.get("act_out", "").lower()

                if Xtr and not Ytr:
                    Xtr, Ytr = split_xy_if_needed(Xtr)
                if Xte and not Yte:
                    Xte, Yte = split_xy_if_needed(Xte)

                Xtr = np.array(Xtr, dtype=np.float64)
                ytr = np.array(Ytr, dtype=np.float64).reshape(-1)
                Xte = np.array(Xte, dtype=np.float64) if Xte else None
                yte = np.array(Yte, dtype=np.float64).reshape(-1) if Yte else None

                if STATE["sizes"] is None:
                    n_in = Xtr.shape[1]
                    classes = int(np.max(ytr)) + 1
                    n_out = 1 if classes == 2 else classes
                    sizes = [n_in, max(8, n_in*2), n_out]
                    ensure_model(sizes)

                classes = int(np.max(ytr)) + 1
                act_out = "sigmoid" if (STATE["sizes"][-1] == 1) else "softmax"
                if STATE["sizes"][-1] > 1: act_out = "softmax"
                STATE["act_hidden"] = act_hidden
                STATE["act_out"] = act_out

                mu, std = zscore_fit(Xtr); STATE["mu"], STATE["std"] = mu, std
                Xtr_n = zscore_apply(Xtr, mu, std)
                Xte_n = zscore_apply(Xte, mu, std) if Xte is not None else None

                if STATE["act_out"] == "sigmoid":
                    Ytr = ytr.reshape(-1,1)
                else:
                    Ytr = one_hot(ytr.astype(int), STATE["sizes"][-1] if STATE["sizes"][-1]>1 else classes)

                batch = max(1, min(64, Xtr_n.shape[0]//4))
                steps = max(1, Xtr_n.shape[0] // batch)

                for ep in range(1, epochs+1):
                    idx = np.random.permutation(Xtr_n.shape[0])
                    Xs = Xtr_n[idx]; Ys = Ytr[idx]
                    for s in range(steps):
                        xb = Xs[s*batch:(s+1)*batch]
                        yb = Ys[s*batch:(s+1)*batch]
                        Z, A = forward(xb)
                        if STATE["act_out"] == "sigmoid":
                            loss, dZL = bce_sigmoid_grad(A[-1], yb)
                        else:
                            loss, dZL = ce_softmax_grad(A[-1], yb)
                        dW = [None]*len(STATE["W"])
                        db = [None]*len(STATE["b"])

                        dA = dZL
                        dW[-1] = A[-2].T @ dA / xb.shape[0]
                        db[-1] = np.mean(dA, axis=0, keepdims=True)

                        for l in range(len(STATE["W"]) - 2, -1, -1):
                            dA_prev = dA @ STATE["W"][l+1].T
                            dZ = dA_prev * dactivation(A[l+1], STATE["act_hidden"])
                            dW[l] = A[l].T @ dZ / xb.shape[0]
                            db[l] = np.mean(dZ, axis=0, keepdims=True)
                            dA = dZ

                        for l in range(len(STATE["W"])):
                            STATE["W"][l] -= lr * dW[l]
                            STATE["b"][l] -= lr * db[l]

                    ytr_pred = predict_classes(Xtr_n)
                    acc_tr = accuracy(ytr.astype(int), ytr_pred.astype(int))
                    acc_te = None
                    if Xte_n is not None and yte is not None and yte.size > 0:
                        yte_pred = predict_classes(Xte_n)
                        acc_te = accuracy(yte.astype(int), yte_pred.astype(int))
                    STATE["history"].append({"epoch": len(STATE["history"])+1, "train_acc":acc_tr, "test_acc":acc_te})

                self._ok({"history": STATE["history"]}); return

            if self.path == "/api/mlp/eval_one":
                x = np.array(data.get("x", []), dtype=np.float64).reshape(1,-1)
                mu, std = STATE["mu"], STATE["std"]
                x = zscore_apply(x, mu, std) if (mu is not None and std is not None) else x
                _, A = forward(x)
                y = A[-1].reshape(-1).tolist()
                self._ok({"y": y}); return

            if self.path == "/api/mlp/eval_file":
                rows = data.get("rows", [])
                out = []
                for i, r in enumerate(rows, start=1):
                    try:
                        x = np.array(r, dtype=np.float64).reshape(1,-1)
                        mu, std = STATE["mu"], STATE["std"]
                        x = zscore_apply(x, mu, std) if (mu is not None and std is not None) else x
                        _, A = forward(x)
                        y = A[-1].reshape(-1).tolist()
                        out.append({"idx": i, "x": r, "y": y, "err": ""})
                    except Exception as e:
                        out.append({"idx": i, "x": r, "y": None, "err": str(e)})
                self._ok({"results": out}); return

            if self.path == "/api/mlp/save":
                obj = save_config_json()
                self._ok(obj); return

            self.send_response(404); self.end_headers(); self.wfile.write(b'{"error":"Not Found"}')
        except Exception as e:
            self.send_response(400); self.end_headers(); self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

    def _ok(self, payload):
        self.send_response(200)
        self.send_header("Content-Type","application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

def main(port=9000):
    h = partial(Handler, directory=str(WEB_ROOT))
    httpd = ThreadingHTTPServer(("127.0.0.1", port), h)
    print(f"Serving {WEB_ROOT} at http://127.0.0.1:{port}{INDEX_URL}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()