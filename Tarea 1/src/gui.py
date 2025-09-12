import csv, math
import matplotlib.pyplot as plt

def load_xy(path):
    X, Y = [], []
    with open(path, 'r', encoding='utf-8-sig') as f:
        rdr = csv.reader(f)
        rows = [ [c.strip() for c in r if c.strip()!=''] for r in rdr if any(c.strip() for c in r) ]
    if not rows: raise ValueError("CSV vacío.")
    def try_row(r):
        try: vals = [float(c) for c in r]; return vals, True
        except: return None, False
    vals, ok = try_row(rows[0])
    start = 1 if not ok else 0
    for i in range(start, len(rows)):
        vals, ok = try_row(rows[i])
        if not ok or len(vals) < 2:
            raise ValueError(f"Fila {i+1}: formato inválido (se requieren n-1 entradas + 1 salida).")
        X.append(vals[:-1]); Y.append(vals[-1])
    m = len(X[0])
    if any(len(r)!=m for r in X): raise ValueError("Columnas inconsistentes.")
    return X, Y

def step(v): return 1.0 if v >= 0.0 else 0.0
def sigmoid(v):
    if v >= 0: z = math.exp(-v); return 1.0/(1.0+z)
    else:      z = math.exp(v);  return z/(1.0+z)

def act(v, kind):
    k = (kind or '').lower()
    if k in ('escalon','step'):   return step(v)
    if k in ('sigmoide','sigmoid'): return sigmoid(v)
    raise ValueError("Activación no soportada: use 'escalon' o 'sigmoide'.")

def predict(X, w, b, kind):
    y = []
    for x in X:
        if len(x)!=len(w): raise ValueError("Dimensión de x no coincide con pesos.")
        s = b
        for i in range(len(x)): s += x[i]*w[i]
        y.append(act(s, kind))
    return y

def cls(v): return 1.0 if v>=0.5 else 0.0

def to2d(X):
    if len(X[0])>=2: return [r[:2] for r in X], ('x1','x2')
    if len(X[0])==1: return [[r[0],0.0] for r in X], ('x1','0')
    raise ValueError("No graficable.")

def scatter(X2, colors, title, xl, yl):
    xs = [r[0] for r in X2]; ys = [r[1] for r in X2]
    plt.figure(); plt.scatter(xs, ys, c=colors); plt.xlabel(xl); plt.ylabel(yl); plt.title(title)

def ask_float(msg):
    while True:
        try: return float(input(msg).strip())
        except: print("Número inválido.")

def ask_weights(n):
    while True:
        parts = [p.strip() for p in input(f"Ingrese {n} pesos separados por coma: ").split(',') if p.strip()!='']
        if len(parts)!=n: print(f"Se esperaban {n} valores."); continue
        try: return [float(p) for p in parts]
        except: print("Todos los pesos deben ser numéricos.")

def ask_act():
    while True:
        k = input("Función de activación ('escalon' o 'sigmoide'): ").strip().lower()
        if k in ('escalon','sigmoide','step','sigmoid'): return k
        print("Opción inválida.")

def main():
    print("=== Tarea 1 – Perceptrón (consola) ===")
    path = input("Ruta del CSV (p.ej., fuzzy_separables.csv o no_separables.csv): ").strip()
    X, Y = load_xy(path)
    n = len(X[0])
    print(f"Cargado: {len(X)} filas | entradas={n} | salida=última columna")

    while True:
        k  = ask_act()
        b  = ask_float("Bias (b): ")
        w  = ask_weights(n)
        yp = predict(X, w, b, k)

        hit_colors = ['green' if cls(t)==cls(p) else 'red' for t,p in zip(Y, yp)]
        X2,(xl,yl) = to2d(X)
        suf = " (x1 vs x2)" if n>2 else ""

        colors_exp = ['blue' if cls(t)==1 else 'orange' for t in Y]

        colors_pred = ['blue' if cls(p)==1 else 'orange' for p in yp]

        colors_match = ['green' if cls(t)==cls(p) else 'red' for t,p in zip(Y, yp)]

        scatter(X2, colors_exp,   f"Esperado{suf}", xl, yl)
        scatter(X2, colors_pred,  f"Predicho{suf}", xl, yl)
        scatter(X2, colors_match, f"Coincidencia (verde=ok, rojo=fail){suf}", xl, yl)

        print("Cierra las 3 figuras para continuar…")
        plt.show()
        if input("¿Probar otros pesos? (s/n): ").strip().lower()!='s': break

if __name__ == "__main__":
    main()