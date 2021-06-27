import sympy as sy
import numpy as np

x1, x2, x3, x4, x5, x6, x7, x8, x9 = sy.symbols('x1, x2, x3, x4, x5, x6, x7, x8, x9')


# Abbruchkriterium a): Abbruch nach einer bestimmten Anzahl Iterationen
def is_finished_max_iterations(f, x, n_max):
    return x.shape[0] - 1 >= n_max


# Abbruchkriterium b): Abbruch, wenn ‖x(n+1) - x(n)‖₂ ≤ ‖x(n+1)‖₂ * 𝛜
def is_finished_relative_error(f, x, eps):
    if x.shape[0] < 2:
        return False

    return np.linalg.norm(x[-1] - x[-2], 2) <= np.linalg.norm(x[-1], 2) * 1.0 * eps


# Abbruchkriterium c): Abbruch, wenn ‖x(n+1) - x(n)‖₂ ≤ 𝛜
def is_finished_absolute_error(f, x, eps):
    if x.shape[0] < 2:
        return False

    return np.linalg.norm(x[-1] - x[-2], 2) <= 1.0 * eps


# Abbruchkriterium d): Abbruch, wenn ‖f(x(n+1))‖₂ ≤ 𝛜
def is_finished_max_residual(f, x, eps):
    if x.shape[0] < 1:
        return False

    return np.linalg.norm(f(x[-1]), 2) <= 1.0 * eps


"""
=======================================================================================================================
INPUT
=======================================================================================================================
"""

# ACHTUNG: Für sinus/cosinus/Exponentialfunktion immer sy.sin/sy.cos/sy.exp/sy.ln/sy.abs verwenden!
f = sy.Matrix([
    x1 ** 2 / 186 ** 2 - x2 ** 2 / (300 ** 2 - 186 ** 2) - 1,
    (x2 - 500) ** 2 / 279 ** 2 - (x1 - 300) ** 2 / (500 ** 2 - 279 ** 2) - 1
])

x = sy.Matrix([x1, x2])       # Wenn mehr oder weniger als 2 Variablen auftreten, diese Liste anpassen!
x0 = np.array([-1000, 900])  # Startwert

k_max = 4  # Maximale Alternativen für 𝛅 (vgl. Skript Seite 107)


# Wähle das Abbruchkriterium (bei passender Zeile Kommentar entfernen):
def is_finished(f, x):
    return is_finished_max_iterations(f, x, 9)      # Abbruchkriterium a): Abbruch nach einer bestimmten Anzahl Iterationen
    # return is_finished_relative_error(f, x, 1e-5)  # Abbruchkriterium b): Abbruch, wenn ‖x(n+1) - x(n)‖₂ ≤ ‖x(n+1)‖₂ * 𝛜
    # return is_finished_absolute_error(f, x, 1e-5)  # Abbruchkriterium c): Abbruch, wenn ‖x(n+1) - x(n)‖₂ ≤ 𝛜
    # return is_finished_max_residual(f, x, 1e-5)    # Abbruchkriterium d): Abbruch, wenn ‖f(x(n+1))‖₂ ≤ 𝛜


"""
=======================================================================================================================
"""

# Bilde die allgemeine Jacobi-Matrix
Df = f.jacobian(x)

print('Ganze Jacobi-Matrix: Df = ' + str(Df))
print('LATEX (Zum Anschauen eingeben unter https://www.codecogs.com/latex/eqneditor.php):')
print(sy.latex(Df))
print('Für eine schrittweise, detaillierte Berechnung der Jacobi-Matrix kann das Skript "5_2_4_Jacobi_Matrix_schrittweise_von_Hand_berechnen.py" verwendet werden')
print()

# Sympy-Funktionen kompatibel mit Numpy machen
f_lambda = sy.lambdify([(x1, x2)], f, "numpy")
Df_lambda = sy.lambdify([(x1, x2)], Df, "numpy")

# Newton-Iterationen
x_approx = np.empty(shape=(0, 2), dtype=np.float64)  # Array mit Lösungsvektoren x0 bis xn
x_approx = np.append(x_approx, [x0], axis=0)  # Start-Vektor in Array einfügen
print('x({}) = {}\n'.format(0, x0))

while not is_finished(f_lambda, x_approx):
    i = x_approx.shape[0] - 1
    print('ITERATION ' + str(i + 1))
    print('-------------------------------------')

    x_n = x_approx[-1]  # x(n) (letzter berechneter Wert)

    print('𝛅({}) ist die Lösung des LGS Df(x({})) * 𝛅({}) = -1 * f(x({}))'.format(i, i, i, i))
    print('Df(x({})) = \n{},\nf(x({})) = \n{}'.format(i, Df_lambda(x_n), i, f_lambda(x_n)))
    print('LGS mit LATEX (Zum Anschauen eingeben unter https://www.codecogs.com/latex/eqneditor.php):')
    print(sy.latex(sy.Matrix(Df_lambda(x_n))) + '\\cdot\\delta^{(' + str(i) + ')}=-1\\cdot' + sy.latex(sy.Matrix(f_lambda(x_n))))

    [Q, R] = np.linalg.qr(Df_lambda(x_n))
    delta = np.linalg.solve(R, -Q.T @ f_lambda(x_n)).flatten()  # 𝛅(n) aus Df(x(n)) * 𝛅(n) = -1 * f(x(n))
    print('𝛅({}) = \n{}\n'.format(i, delta))

    x_next = x_n + delta.reshape(x0.shape[0], )  # x(n+1) = x(n) + 𝛅(n) (provisorischer Kandidat, falls Dämpfung nichts nützt)

    # Finde das minimale k ∈ {0, 1, ..., k_max} für welches 𝛅(n) / 2^k eine verbessernde Lösung ist (vgl. Skript S. 107)
    last_residual = np.linalg.norm(f_lambda(x_n), 2)  # ‖f(x(n))‖₂
    print('Berechne das Residuum der letzten Iteration ‖f(x(n))‖₂ = ' + str(last_residual))

    k = 0
    k_actual = 0
    while k <= k_max:
        print('Versuche es mit k = ' + str(k) + ':')
        new_residual = np.linalg.norm(f_lambda(x_n + (delta.reshape(x0.shape[0], ) / (2 ** k))), 2)  # ‖f(x(n) + 𝛅(n) / 2^k)‖₂
        print('Berechne das neue Residuum ‖f(x(n) + 𝛅(n) / 2^k)‖₂ = ' + str(new_residual))

        if new_residual < last_residual:
            print('Das neue Residuum ist kleiner, verwende also k = ' + str(k))

            delta = delta / (2**k)
            print('𝛅({}) = 𝛅({}) / 2^{} = {}'.format(i, i, k, delta.T))

            x_next = x_n + delta.reshape(x0.shape[0], )  # x(n+1) = x(n) + 𝛅(n) / 2^k
            print('x({}) = x({}) + 𝛅({})'.format(i + 1, i, i))

            k_actual = k
            break  # Minimales k, für welches das Residuum besser ist wurde gefunden -> abbrechen
        else:
            print('Das neue Residuum ist grösser oder gleich gross, versuche ein anderes k (bzw. k = 0 wenn k_max erreicht ist)')

        print()
        k += 1

    x_approx = np.append(x_approx, [x_next], axis=0)

    print('x({}) = {} (k = {})'.format(x_approx.shape[0] - 1, x_next, k_actual))
    print('‖f(x({}))‖₂ = {}'.format(i + 1, np.linalg.norm(f_lambda(x_next), 2)))
    print('‖x({}) - x({})‖₂ = {}\n'.format(i + 1, i, np.linalg.norm(x_next - x_n, 2)))

print(x_approx)
