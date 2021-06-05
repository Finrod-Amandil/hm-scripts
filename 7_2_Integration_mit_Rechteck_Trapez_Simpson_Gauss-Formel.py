import numpy as np
import scipy.integrate


def Rf(f, a, b, n):
    xi = np.array([a + i * ((b - a) / n) for i in range(0, n)], dtype=np.float64)
    h = (b - a) / n
    return h * np.sum(f(xi + (h / 2)))


def Tf(f, a, b, n):
    xi = np.array([a + i * ((b - a) / n) for i in range(1, n)], dtype=np.float64)
    h = (b - a) / n
    return h * ((f(a) + f(b)) / 2 + np.sum(f(xi)))


def Sf(f, a, b, n):
    xi_1_n_m1 = np.array([a + i * ((b - a) / n) for i in range(1, n)], dtype=np.float64)            # xi from 1 to n-1
    xi_1_n = np.array([a + i * ((b - a) / n) for i in range(1, n + 1)], dtype=np.float64)           # xi from 1 to n
    xi_m1_1_n = np.array([a + (i - 1) * ((b - a) / n) for i in range(1, n + 1)], dtype=np.float64)  # x(i-1) from 1 to n
    h = (b - a) / n
    return (h / 3) * ((1 / 2) * f(a) + np.sum(f(xi_1_n_m1)) + 2 * np.sum(f((xi_m1_1_n + xi_1_n) / 2)) + (1 / 2) * f(b))


# Only supported for n = 1, 2, 3
def Gauss_Formula(f, a, b, n):
    if n == 1:
        return (b - a) * f((b + a) / 2.0)
    if n == 2:
        return ((b - a) / 2.0) * (f((-1 / np.sqrt(3.0)) * ((b - a) / 2.0) + ((b + a) / 2.0)) + f((1 / np.sqrt(3.0)) * ((b - a) / 2.0) + ((b + a) / 2.0)))
    if n == 3:
        return ((b - a) / 2.0) * ((5.0 / 9.0) * f(-np.sqrt(0.6) * ((b - a) / 2.0) + ((b + a) / 2.0)) + (8.0 / 9.0) * f((b + a) / 2.0) + (5.0 / 9.0) * f(np.sqrt(0.6) * ((b - a) / 2.0) + ((b + a) / 2.0)))
    return 0


# Zu integrierende Funktion
def f(x):
    return np.log(x ** 2)


tol = 1e-5
a = 1
b = 2

integral, _ = scipy.integrate.quad(f, 1, 2, epsabs=1e-10)

# --------------------------------------------------------------------
# Rechteck-Regel
n = 0
error = tol + 1
while error > tol:
    n += 1
    error = np.abs(integral - Rf(f, a, b, n))

print('RECHTECK-REGEL')
print('Wert des Integrals (scipy) = ' + str(integral))
print('Wert mit Rechteckregel     = ' + str(Rf(f, a, b, n)))
print('Absoluter Fehler = ' + str(error))
print('n = ' + str(n))
print('h = ' + str((b - a) / n))
print('-----------------------------------------------')

# --------------------------------------------------------------------
# Trapez-Regel
n = 0
error = tol + 1
while error > tol:
    n += 1
    error = np.abs(integral - Tf(f, a, b, n))

print('TRAPEZ-REGEL')
print('Wert des Integrals (scipy) = ' + str(integral))
print('Wert mit Trapezregel       = ' + str(Tf(f, a, b, n)))
print('Absoluter Fehler = ' + str(error))
print('n = ' + str(n))
print('h = ' + str((b - a) / n))
print('-----------------------------------------------')

# --------------------------------------------------------------------
# Simpson-Regel
n = 0
error = tol + 1
while error > tol:
    n += 1
    error = np.abs(integral - Sf(f, a, b, n))

print('SIMPSON-REGEL')
print('Wert des Integrals (scipy) = ' + str(integral))
print('Wert mit Simpson-Regel     = ' + str(Sf(f, a, b, n)))
print('Absoluter Fehler = ' + str(error))
print('n = ' + str(n))
print('h = ' + str((b - a) / n))
print('-----------------------------------------------')

# --------------------------------------------------------------------
# Gauss-Formeln
for n in range(1, 4):
    error = np.abs(integral - Gauss_Formula(f, a, b, n))
    print('GAUSS-FORMEL mit n = {}'.format(n))
    print('Wert des Integrals (scipy) = ' + str(integral))
    print('Wert mit Gauss-Formel (n = {}) = '.format(n) + str(Gauss_Formula(f, a, b, n)))
    print('Absoluter Fehler = ' + str(error))
    print()
