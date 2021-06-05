import sympy as sy
import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 1, 2, 3, 4], dtype=np.float64)  # Messwerte xi
y = np.array([3, 1, 0.5, 0.2, 0.05], dtype=np.float64)  # Messwerte yi


def f(x, a):
    return a[0] * sy.exp(a[1] * x)


lam0 = np.array([3, -1], dtype=np.float64)  # Startvektor für Iteration
a = sy.symbols('a:{n:d}'.format(n=lam0.size))

tol = 1e-5     # Fehlertoleranz
max_iter = 30  # Maximale Iterationen
pmax = 4       # Maximale Dämpfung

g = sy.Matrix([y[k] - f(x[k], a) for k in range(x.shape[0])])  # Fehlerfunktion für alle (xi, yi)
Dg = g.jacobian(a)

g_lambda = sy.lambdify([a], g, 'numpy')
Dg_lambda = sy.lambdify([a], Dg, 'numpy')

k = 0
lam = np.copy(lam0)
increment = tol + 1
err_func = np.linalg.norm(g_lambda(lam)) ** 2

while increment > tol and k <= max_iter:
    # QR-Zerlegung von Dg(lam)
    [Q, R] = np.linalg.qr(Dg_lambda(lam))
    delta = np.linalg.solve(R, -Q.T @ g_lambda(lam)).flatten()
    # Achtung: flatten() braucht es, um aus dem Spaltenvektor delta wieder
    # eine "flachen" Vektor zu machen, da g hier nicht mit Spaltenvektoren als Input umgehen kann

    # Dämpfung
    p = 0
    while p <= pmax and np.linalg.norm(g_lambda(lam + delta / (2 ** p))) ** 2 >= np.linalg.norm(g_lambda(lam)) ** 2:
        p += 1

    if p > pmax:
        p = 0

    # Update des Vektors Lambda
    lam = lam + delta / (2 ** p)

    err_func = np.linalg.norm(g_lambda(lam)) ** 2
    increment = np.linalg.norm(delta)
    k = k + 1
    print('Iteration: ', k)
    print('lambda = ', lam)
    print('Inkrement = ', increment)
    print('Fehlerfunktional =', err_func)
    print()
