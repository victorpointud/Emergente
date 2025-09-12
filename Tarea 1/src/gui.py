import math
import matplotlib.pyplot as plt

def read_csv_xy(path):
    X, Y = [], []

    def try_float(s: str):
        s = s.strip()
        if s == "":
            raise ValueError("empty")
        try:
            return float(s)
        except ValueError:
            return None

    with open(path, 'r', encoding='utf-8-sig') as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if not lines:
        raise ValueError("El CSV está vacío.")

    def parse_row(line: str):
        parts = [p.strip() for p in line.split(',') if p.strip() != ""]
        vals = [try_float(p) for p in parts]
        return parts, vals
    
    first_parts, first_vals = parse_row(lines[0])
    has_header = any(v is None for v in first_vals)

    start_idx = 1 if has_header else 0
    for i, line in enumerate(lines[start_idx:], start=start_idx + 1):
        parts, vals = parse_row(line)
        if any(v is None for v in vals):
            raise ValueError(f"Fila {i}: valor no numérico. Revise separadores/comas.")
        if len(vals) < 2:
            raise ValueError(f"Fila {i}: se requieren ≥2 columnas (n-1 entradas + 1 salida).")
        X.append([float(v) for v in vals[:-1]])
        Y.append(float(vals[-1]))

    m = len(X[0])
    for i, row in enumerate(X, start=1):
        if len(row) != m:
            raise ValueError(f"Fila {i+start_idx}: número de columnas inconsistente.")

    return X, Y

def weighted_sum(x, w, b):
    if len(x) != len(w):
        raise ValueError(f"Dimensión incompatible: len(x)={len(x)} vs len(w)={len(w)}")
    s = b
    for i in range(len(x)):
        s += x[i] * w[i]
    return s

def activation(v, kind):
    k = (kind or '').strip().lower()
    if k in ('escalon', 'step'):
        return 1.0 if v >= 0.0 else 0.0
    if k in ('sigmoide', 'sigmoid'):
        if v >= 0:
            z = math.exp(-v)
            return 1.0 / (1.0 + z)
        else:
            z = math.exp(v)
            return z / (1.0 + z)
    raise ValueError("Función de activación no soportada. Use 'escalon' o 'sigmoide'.")

def predict_perceptron(X, w, b, act):
    y_pred = []
    for x in X:
        z = weighted_sum(x, w, b)
        y_pred.append(activation(z, act))
    return y_pred

def to_class(v):
    return 1.0 if v >= 0.5 else 0.0

def match_colors(y_true, y_pred):
    cols, hits = [], []
    for t, p in zip(y_true, y_pred):
        ok = (to_class(t) == to_class(p))
        hits.append(ok)
        cols.append('green' if ok else 'red')
    return cols, hits

def ensure_2d(X):
    if len(X[0]) >= 2:
        return [row[:2] for row in X], ('x1', 'x2')
    if len(X[0]) == 1:
        return [[row[0], 0.0] for row in X], ('x1', '0')
    raise ValueError("Conjunto de entradas no graficable.")

def scatter_plot(X2, colors, title, xlabel, ylabel):
    xs = [r[0] for r in X2]
    ys = [r[1] for r in X2]
    plt.figure()
    plt.scatter(xs, ys, c=colors)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

def ask_float(prompt):
    while True:
        s = input(prompt).strip()
        try: return float(s)
        except: print("Ingrese un número válido.")

def ask_weights(n):
    while True:
        s = input(f"Ingrese {n} pesos separados por coma: ").strip()
        parts = [p.strip() for p in s.split(',') if p.strip()!='']
        if len(parts) != n:
            print(f"Se esperaban {n} valores. Intente de nuevo.")
            continue
        try: return [float(p) for p in parts]
        except: print("Todos los pesos deben ser numéricos. Intente de nuevo.")

def ask_activation():
    while True:
        k = input("Función de activación ('escalon' o 'sigmoide'): ").strip().lower()
        if k in ('escalon', 'sigmoide', 'step', 'sigmoid'):
            return k
        print("Opción inválida. Use 'escalon' o 'sigmoide'.")

def main():
    print("=== Tarea 1 – Perceptrón (consola) ===")
    csv_path = input("Ruta del CSV (p.ej., fuzzy_separables.csv o no_separables.csv): ").strip()
    X, Y = read_csv_xy(csv_path)
    n_in = len(X[0])
    print(f"Datos: {len(X)} filas | entradas={n_in} | salida=última columna")

    while True:
        act = ask_activation()
        b = ask_float("Bias (b): ")
        w = ask_weights(n_in)

        y_pred = predict_perceptron(X, w, b, act)
        colors, hits = match_colors(Y, y_pred)
        X2, (xl, yl) = ensure_2d(X)
        suf = "(mostrando x1 vs x2)" if n_in > 2 else ""

        scatter_plot(X2, colors, f"Esperado {suf}".strip(), xl, yl)
        scatter_plot(X2, colors, f"Predicho {suf}".strip(), xl, yl)
        scatter_plot(X2, ['green' if h else 'red' for h in hits], f"Coincidencia (verde=ok, rojo=fail) {suf}".strip(), xl, yl)

        print("Cierra las 3 figuras para continuar…")
        plt.show()

        again = input("¿Probar otros pesos? (s/n): ").strip().lower()
        if again != 's':
            break

if __name__ == "__main__":
    main()