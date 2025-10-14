import io
import json
import time
import glob
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from functools import partial
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT  
INDEX_URL = "/templates/pytorch.html"
DATA_DIR = ROOT.parent / "data"
MODELS_DIR = ROOT.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


STATE = {
    "cfg": None,             
    "model": None,          
    "num_classes": None,
    "class_names": None,
    "train_loader": None,
    "test_loader": None,
    "history": [],          
    "last_confusion": None,  
}

FASHION10 = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

FOUR_CLASS_NAMES = ["Top", "Bottom", "Footwear", "Bag"]
FASHION10_TO_4 = {
    0: 0, 
    1: 1,  
    2: 0,  
    3: 0,  
    4: 0, 
    5: 2,  
    6: 0, 
    7: 2,  
    8: 3, 
    9: 2,  
}

class MapTo4Classes(Dataset):
    def __init__(self, base_ds):
        self.base = base_ds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, y = self.base[idx]
        return img, FASHION10_TO_4[int(y)]

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.0):
        super().__init__()
        dims = [input_dim] + hidden_dims + [num_classes]
        layers = []
        for i in range(len(dims)-2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.net(x)

def linspace_int(start, end, steps):
    arr = np.linspace(start, end, steps)
    return [max(16, int(round(v))) for v in arr]

def build_hidden_dims(shape, n_hidden, width, input_dim, num_classes):
    if n_hidden <= 0:
        return []
    if shape == "rectangular":
        return [max(16, int(width))] * n_hidden
    pts = linspace_int(input_dim, max(num_classes, 8), n_hidden + 2)[1:-1]
    return pts

def accuracy_from_logits(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()

def compute_confusion(logits, targets, C):
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    y = targets.cpu().numpy()
    cm = np.zeros((C, C), dtype=int)
    for t, p in zip(y, preds):
        cm[int(t), int(p)] += 1
    return cm

def history_plot_png(history):
    if not history:
        return None
    epochs = [h["epoch"] for h in history]
    tr_loss = [h["train_loss"] for h in history]
    te_acc  = [h["test_acc"] for h in history]

    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(epochs, tr_loss, label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax2 = ax1.twinx()
    ax2.plot(epochs, te_acc, label="test acc", linestyle="--")
    ax2.set_ylabel("acc")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    plt.title("Training history")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def confusion_plot_png(cm, class_names):
    if cm is None:
        return None
    plt.clf()
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="black")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def latest_model_path():
    files = sorted(glob.glob(str(MODELS_DIR / "pytorch_mlp_*.pth")))
    return files[-1] if files else None

def make_dataloaders(four_classes=False, batch_size=128):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_base = datasets.FashionMNIST(str(DATA_DIR), train=True, download=True, transform=tfm)
    test_base  = datasets.FashionMNIST(str(DATA_DIR), train=False, download=True, transform=tfm)

    if four_classes:
        train_ds = MapTo4Classes(train_base)
        test_ds  = MapTo4Classes(test_base)
        class_names = FOUR_CLASS_NAMES
        num_classes = 4
    else:
        train_ds = train_base
        test_ds  = test_base
        class_names = FASHION10
        num_classes = 10

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader, num_classes, class_names

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running = 0.0
    total = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        bs = images.size(0)
        running += loss.item() * bs
        total += bs
    return running / max(1, total)

@torch.no_grad()
def evaluate(model, loader, criterion, num_classes):
    model.eval()
    total = 0
    correct = 0
    running = 0.0
    all_logits = []
    all_targets = []
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)
        loss = criterion(logits, labels)
        running += loss.item() * images.size(0)
        total += images.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        all_logits.append(logits.detach().cpu())
        all_targets.append(labels.detach().cpu())
    avg_loss = running / max(1, total)
    acc = correct / max(1, total)
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    cm = compute_confusion(all_logits, all_targets, num_classes)
    return avg_loss, acc, cm

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/plot/loss.png":
            png = history_plot_png(STATE["history"])
            if png is None:
                self.send_response(404); self.end_headers(); return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.end_headers()
            self.wfile.write(png)
            return

        if p == "/plot/conf.png":
            png = confusion_plot_png(STATE["last_confusion"], STATE.get("class_names") or [])
            if png is None:
                self.send_response(404); self.end_headers(); return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.end_headers()
            self.wfile.write(png)
            return

        return super().do_GET()

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length","0"))
            raw = self.rfile.read(n).decode("utf-8")
            data = json.loads(raw) if raw else {}
        except Exception:
            self.send_response(400); self.end_headers(); self.wfile.write(b'{"error":"Invalid JSON"}'); return

        try:
            if self.path == "/api/create":
                self._api_create(data); return
            if self.path == "/api/train":
                self._api_train(data); return
            if self.path == "/api/save":
                self._api_save(data); return
            if self.path == "/api/load":
                self._api_load(data); return

            self.send_response(404); self.end_headers(); self.wfile.write(b'{"error":"Not Found"}')
        except Exception as e:
            self.send_response(400); self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

    def _api_create(self, data):
        n_hidden   = int(data.get("n_hidden", 2))
        shape      = str(data.get("shape", "rectangular")).lower()
        width      = int(data.get("width", 512))
        dropout    = float(data.get("dropout", 0.2))
        epochs     = int(data.get("epochs", 5))
        lr         = float(data.get("lr", 1e-3))
        optimizer  = str(data.get("optimizer", "adam")).lower()
        batch_size = int(data.get("batch_size", 128))
        four_cls   = bool(data.get("four_classes", False))

        train_loader, test_loader, num_classes, class_names = make_dataloaders(four_classes=four_cls, batch_size=batch_size)

        input_dim = 28*28
        hidden_dims = build_hidden_dims(shape, n_hidden, width, input_dim, num_classes)
        model = MLP(input_dim, hidden_dims, num_classes, dropout=dropout).to(DEVICE)

        STATE["cfg"] = {
            "n_hidden": n_hidden, "shape": shape, "width": width,
            "dropout": dropout, "epochs": epochs, "lr": lr,
            "optimizer": optimizer, "batch_size": batch_size,
            "four_classes": four_cls
        }
        STATE["model"] = model
        STATE["num_classes"] = num_classes
        STATE["class_names"] = class_names
        STATE["train_loader"] = train_loader
        STATE["test_loader"] = test_loader
        STATE["history"].clear()
        STATE["last_confusion"] = None

        summary = f"MLP: input=784, hidden={hidden_dims}, classes={num_classes} ({'4 clases' if four_cls else '10 clases'}), device={DEVICE}"
        self._ok({"summary": summary})

    def _api_train(self, _data):
        if STATE["model"] is None or STATE["train_loader"] is None:
            raise ValueError("Primero crea la red (API /api/create).")

        cfg = STATE["cfg"]
        model = STATE["model"]
        train_loader = STATE["train_loader"]
        test_loader  = STATE["test_loader"]
        num_classes  = STATE["num_classes"]

        criterion = nn.CrossEntropyLoss()
        if cfg["optimizer"] == "sgd":
            opt = optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9)
        else:
            opt = optim.Adam(model.parameters(), lr=cfg["lr"])

        E = max(1, int(cfg["epochs"]))
        STATE["history"].clear()
        for ep in range(1, E+1):
            tr_loss = train_one_epoch(model, train_loader, criterion, opt)
            te_loss, te_acc, cm = evaluate(model, test_loader, criterion, num_classes)
            STATE["history"].append({"epoch": ep, "train_loss": float(tr_loss), "test_acc": float(te_acc)})
            STATE["last_confusion"] = cm

        rows = []
        rows.append("<table><thead><tr><th>Epoch</th><th>Train Loss</th><th>Test Acc</th></tr></thead><tbody>")
        for h in STATE["history"]:
            rows.append(f"<tr><td>{h['epoch']}</td><td>{h['train_loss']:.4f}</td><td>{h['test_acc']:.4f}</td></tr>")
        rows.append("</tbody></table>")
        log_html = "".join(rows)
        self._ok({"log_html": log_html})

    def _api_save(self, _data):
        if STATE["model"] is None:
            raise ValueError("No hay modelo en memoria.")
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = MODELS_DIR / f"pytorch_mlp_{ts}.pth"
        payload = {
            "state_dict": STATE["model"].state_dict(),
            "cfg": STATE["cfg"],
            "class_names": STATE["class_names"],
            "num_classes": STATE["num_classes"],
        }
        torch.save(payload, path)
        self._ok({"path": str(path)})

    def _api_load(self, _data):
        p = latest_model_path()
        if p is None:
            raise FileNotFoundError("No se encontró ningún .pth en /models.")
        blob = torch.load(p, map_location=DEVICE)
        cfg = blob["cfg"]
        class_names = blob["class_names"]
        num_classes = blob["num_classes"]

        train_loader, test_loader, numC, classN = make_dataloaders(
            four_classes=bool(cfg.get("four_classes", False)),
            batch_size=int(cfg.get("batch_size", 128))
        )
        input_dim = 28*28
        hidden_dims = build_hidden_dims(cfg["shape"], int(cfg["n_hidden"]), int(cfg["width"]),
                                        input_dim, num_classes)
        model = MLP(input_dim, hidden_dims, num_classes, dropout=float(cfg["dropout"])).to(DEVICE)
        model.load_state_dict(blob["state_dict"])

        STATE["cfg"] = cfg
        STATE["model"] = model
        STATE["num_classes"] = num_classes
        STATE["class_names"] = class_names
        STATE["train_loader"] = train_loader
        STATE["test_loader"] = test_loader
        STATE["history"].clear()
        STATE["last_confusion"] = None

        summary = f"Cargado {Path(p).name} | hidden={hidden_dims}, classes={num_classes}, device={DEVICE}"
        self._ok({"summary": summary})

    def _ok(self, payload: dict):
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

def main(port=9000):
    h = partial(Handler, directory=str(WEB_ROOT))
    httpd = ThreadingHTTPServer(("127.0.0.1", port), h)
    print(f"Serving {WEB_ROOT} at http://127.0.0.1:{port}{INDEX_URL} (device={DEVICE})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()