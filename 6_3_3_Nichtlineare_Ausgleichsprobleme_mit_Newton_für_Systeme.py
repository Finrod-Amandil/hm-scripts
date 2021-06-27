import numpy as np
import sympy as sy
import matplotlib.pyplot as plt

a1, a2 = sy.symbols('a1, a2')  # Gesuchte Parameter 𝛌1 .. 𝛌m
a = sy.Matrix([a1, a2])
parameter_count = a.shape[0]
x, y = sy.symbols('x, y')

x_m = np.array([0, 1, 2, 3, 4], dtype=np.float64)  # Messwerte xi
y_m = np.array([3, 1, 0.5, 0.2, 0.05], dtype=np.float64)  # Messwerte yi

f = a1 * sy.exp(a2 * x)  # ANSATZFUNKTION (Achtung: Für sin/cos/exp/log die Sympy-Funktionen verwenden!

error_function = y - f  # Differenz zwischen Messwerten yi und berechneten Werten durch Ansatzfunktion f(x, a1, a2...)
error_function_squared = error_function ** 2  # Quadrierte Fehlerfunktion (soll minimal werden)

# Berechne die partiellen Ableitungen für alle 𝛌1 .. 𝛌m (sollen alle 0 werden)
error_function_squared_diff = sy.Matrix([sy.diff(error_function_squared, ai) for ai in a])

# Abgeleitete Fehlerfunktionale ∂E(f)/∂ai = ∑[i = 1 .. n] ∂(yi - f(xi, a1, a2, ...))/∂ai sollen null werden
# Die Ableitungen ∂(yi - f(xi, a1, a2, ...))/∂ai wurden jetzt bereits berechnet, jetzt fehlt noch das einsetzen
# der xi und yi und Summe
error_functional_diff = sy.Matrix([0 for i in range(parameter_count)])  # Jede Zeile enthält später eine Summe
for j in range(parameter_count):
    sum = 0
    for i in range(x_m.shape[0]):
        sum += error_function_squared_diff[j].subs(x, x_m[i]).subs(y, y_m[i])  # Ersetze x und y mit xi und yi
    error_functional_diff[j] = sum


# Das Fehlerfunktional besteht aus m Funktionen mit den Unbekannten a1 .. am (m Unbekannte). Alle Funktionen
# sollen 0 sein => System von LGS kann mit Newton-Verfahren für Systeme gelöst werden.

f = error_functional_diff
lam = a  # a1, a2, .... am
Df = f.jacobian(lam)  # Jacobi-Matrix für Fehlerfunktional

# Werte für Newton-Iterationen vorbereiten
f_lambda = sy.lambdify([(a1, a2)], f, "numpy")
Df_lambda = sy.lambdify([(a1, a2)], Df, "numpy")
x0 = np.array([3, -1], dtype=np.float64)  # Startwert für die Iteration
eps = 1e-5  # Fehlertoleranz

# Newton-Iterationen (Standard, für vereinfachtes und gedämpftes Verfahren vgl. Skripte 5_4_x)
x_approx = np.empty(shape=(0, 2), dtype=np.float64)  # Array mit Lösungsvektoren x0 bis xn
x_approx = np.append(x_approx, [x0], axis=0)  # Start-Vektor in Array einfügen

while x_approx.shape[0] < 2 or not np.linalg.norm(x_approx[-1] - x_approx[-2], 2) <= 1.0 * eps:  # Für weitere Abbruchbedingungen vgl. Skripte 5_4_x
    i = x_approx.shape[0] - 1
    x_n = x_approx[-1]  # x(n) (letzter berechneter Wert)
    [Q, R] = np.linalg.qr(Df_lambda(x_n))
    delta = np.linalg.solve(R, -Q.T @ f_lambda(x_n)).flatten()  # 𝛅(n) aus Df(x(n)) * 𝛅(n) = -1 * f(x(n))
    x_next = x_n + delta.reshape(x0.shape[0], )                  # x(n+1) = x(n) + 𝛅(n)
    x_approx = np.append(x_approx, [x_next], axis=0)

print(x_approx)
