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


def f(x, y):
    return ((x ** 2) * 1.0) / y  # Rechte Seite der DGL y' = f(x, y)


def y_exact(x):
    return np.sqrt((2 * x ** 3) / 3.0 + 4)  # Exakte Lösung der DGL


a = 0
b = 1.4
n = 4
y0 = 2
h = ((b - a) * 1.0) / n
x = np.arange(a, b + h, step=h, dtype=np.float64)

y_euler = interpolate_euler(f, x, h, y0)
y_midpoint = interpolate_midpoint(f, x, h, y0)
y_mod_euler = interpolate_mod_euler(f, x, h, y0)

print('x = ' + str(x))
print('y_euler = ' + str(y_euler))
print('y_midpoint = ' + str(y_midpoint))
print('y_mod_euler = ' + str(y_mod_euler))

xmin = -0.1
xmax = 1.5
ymin = 1.9
ymax = 2.6
h_meshgrid = 0.05
x_plot = np.arange(xmin, xmax, step=h_meshgrid, dtype=np.float64)
y_plot = np.arange(ymin, ymax, step=h_meshgrid, dtype=np.float64)
[x_grid, y_grid] = np.meshgrid(x_plot, y_plot)

x_exact = np.arange(xmin, xmax, step=0.001, dtype=np.float64)

dy = f(x_grid, y_grid)
dx = np.full((dy.shape[0], dy.shape[1]), 1, dtype=np.float64)

plt.figure(0)
plt.quiver(x_grid, y_grid, dx, dy)
plt.plot(x_exact, y_exact(x_exact), label='Exact')
plt.plot(x, y_euler, label='Euler')
plt.plot(x, y_midpoint, label='Midpoint')
plt.plot(x, y_mod_euler, label='Mod. Euler')
plt.legend()
plt.show()
