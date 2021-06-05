import numpy as np
import matplotlib.pyplot as plt


def interpolate_runge_kutta(f, x, h, y0):
    y = np.full(x.shape[0], 0, dtype=np.float64)
    y[0] = y0

    for i in range(x.shape[0] - 1):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + (h / 2.0), y[i] + (h / 2.0) * k1)
        k3 = f(x[i] + (h / 2.0), y[i] + (h / 2.0) * k2)
        k4 = f(x[i] + h, y[i] + h * k3)

        y[i + 1] = y[i] + h * (1 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return y


def interpolate_runge_kutta_custom(f, x, h, y0):
    s = 4

    a = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0.75, 0.5, 0.75, 0]
    ], dtype=np.float64)
    b = np.array([0.1, 0.1, 0.4, 0.4], dtype=np.float64)
    c = np.array([0.75, 0.25, 0.75, 0.5], dtype=np.float64)

    y = np.full(x.shape[0], 0, dtype=np.float64)
    y[0] = y0

    for i in range(x.shape[0] - 1):
        k = np.full(s, 0, dtype=np.float64)

        for n in range(s):
            k[n] = f(x[i] + (c[n] * h), y[i] + h * np.sum([a[n][m] * k[m] for m in range(n - 1)]))

        y[i + 1] = y[i] + h * np.sum([b[n] * k[n] for n in range(s)])

    return y


def f(x, y):
    return 1 - y / x


def y_exact(x):
    return (x / 2.0) + 9 / (2.0 * x)


a = 1
b = 6
h = 0.01
y0 = 5
x = np.arange(a, b + h, step=h, dtype=np.float64)

y = interpolate_runge_kutta(f, x, h, y0)
y_c = interpolate_runge_kutta_custom(f, x, h, y0)

plt.figure(0)
plt.title('Runge-Kutta vs Runge-Kutta-custom vs Exact')
plt.plot(x, y, label='Runge-Kutta')
plt.plot(x, y_c, label='Runge-Kutta custom')
plt.plot(x, y_exact(x), label='Exact')
plt.legend()
plt.show()

plt.figure(1)
plt.title('Absolute error')
plt.plot(x, np.abs(y - y_exact(x)), label='Runge-Kutta')
plt.plot(x, np.abs(y_c - y_exact(x)), label='Runge-Kutta custom')
plt.legend()
plt.show()
