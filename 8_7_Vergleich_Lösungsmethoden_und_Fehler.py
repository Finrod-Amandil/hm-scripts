import numpy as np
import matplotlib.pyplot as plt


def interpolate_euler(f, x, h, y0):
    y = np.full(x.shape[0], 0, dtype=np.float64)
    y[0] = y0

    for i in range(x.shape[0] - 1):
        y[i + 1] = y[i] + h * f(x[i], y[i])

    return y


def interpolate_midpoint(f, x, h, y0):
    y = np.full(x.shape[0], 0, dtype=np.float64)
    y[0] = y0

    for i in range(x.shape[0] - 1):
        y[i + 1] = y[i] + h * f(x[i] + (h / 2.0), y[i] + (h / 2.0) * f(x[i], y[i]))

    return y


def interpolate_mod_euler(f, x, h, y0):
    y = np.full(x.shape[0], 0, dtype=np.float64)
    y[0] = y0

    for i in range(x.shape[0] - 1):
        y[i + 1] = y[i] + h * (f(x[i], y[i]) + f(x[i + 1], y[i] + h * f(x[i], y[i]))) / 2

    return y


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


def f(x, y):
    return ((x ** 2) * 1.0) / y


def y_exact(x):
    return np.sqrt((2 * x ** 3) / 3.0 + 4)


a = 0
b = 10
h = 0.1
y0 = 2
x = np.arange(a, b + h, step=h, dtype=np.float64)

y_euler = interpolate_euler(f, x, h, y0)
y_midpoint = interpolate_midpoint(f, x, h, y0)
y_mod_euler = interpolate_mod_euler(f, x, h, y0)
y_runge_kutta = interpolate_runge_kutta(f, x, h, y0)

print('x = ' + str(x))
print('y_euler = ' + str(y_euler))
print('y_midpoint = ' + str(y_midpoint))
print('y_mod_euler = ' + str(y_mod_euler))
print('y_runge_kutta = ' + str(y_runge_kutta))

xmin = -0.1
xmax = 1.5
ymin = 1.9
ymax = 2.6

x_exact = np.arange(xmin, xmax, step=0.001, dtype=np.float64)

plt.figure(0)
plt.title('Numerical integration methods')
plt.plot(x_exact, y_exact(x_exact), label='Exact')
plt.plot(x, y_euler, label='Euler')
plt.plot(x, y_midpoint, label='Midpoint')
plt.plot(x, y_mod_euler, label='Mod. Euler')
plt.plot(x, y_runge_kutta, label='Runge-Kutta')
plt.legend()
plt.show()

plt.figure(1)
plt.title('Global error')
plt.semilogy()
plt.plot(x, np.abs(y_euler - y_exact(x)), label='Euler')
plt.plot(x, np.abs(y_midpoint - y_exact(x)), label='Midpoint')
plt.plot(x, np.abs(y_mod_euler - y_exact(x)), label='Mod. Euler')
plt.plot(x, np.abs(y_runge_kutta - y_exact(x)), label='Runge-Kutta')
plt.legend()
plt.show()
