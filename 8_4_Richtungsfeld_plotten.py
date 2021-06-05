import numpy as np
import matplotlib.pyplot as plt


def plot_slope_field(f, xmin, xmax, ymin, ymax, hx, hy):
    x = np.arange(xmin, xmax, step=hx, dtype=np.float64)
    y = np.arange(ymin, ymax, step=hy, dtype=np.float64)
    [x_grid, y_grid] = np.meshgrid(x, y)

    dy = f(x_grid, y_grid)
    dx = np.full((dy.shape[0], dy.shape[1]), 1, dtype=np.float64)

    plt.quiver(x_grid, y_grid, dx, dy)
    plt.show()


def f(x, y):
    return ((x**2) * 1.0) / y  # Rechte Seite der DGL in expliziter Form y' = f(x, y)


plot_slope_field(f, -1, 4, 0.5, 8, 0.5, 0.5)  # f, xmin, xmax, ymin, ymax, hx, hy (Schrittweiten)
