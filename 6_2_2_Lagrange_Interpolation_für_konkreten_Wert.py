import numpy as np


def lagrange_int(x, y, x_int):
    return np.sum([y[i] * np.prod([(x_int - x[j] * 1.0) / (x[i] - x[j]) for j in range(x.shape[0]) if j != i]) for i in range(x.shape[0])])


x = np.array([0, 2500, 5000, 10000])  # xi
y = np.array([1013, 747, 540, 226])   # yi
x_int = 3750  # Zu interpolierende Stelle
y_int = lagrange_int(x, y, x_int)

print("f(" + str(x_int) + ") = " + str(y_int))