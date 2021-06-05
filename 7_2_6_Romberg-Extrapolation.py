import numpy as np


def Tf(f, a, b, n):
    xi = np.array([a + i * ((b - a) / n) for i in range(1, n)], dtype=np.float64)
    h = (b - a) / n
    return h * ((f(a) + f(b)) / 2 + np.sum(f(xi)))


def romberg_extrapolate(f, a, b, m):
    T = np.zeros((m + 1, m + 1), dtype=np.float64)

    T[0:, 0] = [Tf(f, a, b, 2 ** j) for j in range(m + 1)]

    for k in range(1, m + 1):
        for j in range(m + 1 - k):
            T[j, k] = (4**k * T[j + 1, k - 1] - T[j, k - 1]) / (4**k - 1)

    print('Tij = \n', T)
    return T[0, m]


def f(x):
    return np.cos(x ** 2)


a = 0
b = np.pi
m = 4

T = romberg_extrapolate(f, a, b, m)

print('\nIntegral mit Romber-Extrapolation: ', T)
