"""
    Iteratives Lösen eines LGS Ax = b mit Jacobi-Verfahren

    @version: 1.0, 24.01.2021
    @author: zahlesev@students.zhaw.ch
"""

import numpy as np

"""==================== INPUT ===================="""
A = np.array([[8, 5, 2], [5, 9, 1], [4, 2, 7]], dtype=np.float64)
b = np.array([19, 5, 34], dtype=np.float64)
x0 = np.array([1, -1, 3], dtype=np.float64)
tol = 1e-4  # Maximaler a-posteriori Fehler

show_steps = True
"""==============================================="""


def solve_jacobi(A, b, x0, tol):
    if not (A.shape[0] == A.shape[1]):
        raise Exception('A is misshaped.')

    if not (A.shape[0] == b.shape[0]):
        raise Exception('The shape of b does not match the size of A.')

    if not (A.shape[0] == x0.shape[0]):
        raise Exception('The shape of x0 does not match the size of A.')

    B = get_b(A)

    if not (is_converging(B)):
        raise Exception('Iteration will not converge.')
    elif show_steps:
        print("Iteration konvergiert!")

    x = iterate_jacobi(A, B, b, x0, tol)

    n2 = calculate_required_steps_a_priori(B, x0, x[1], tol)

    return x[-1], len(x) - 1, n2


def get_b(A):
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    R = np.triu(A, 1)

    B = -np.linalg.inv(D) @ (L + R)

    if show_steps:
        print("Matrix B berechnen. B = -D⁻¹ * (L + R) = \n" + str(B))

    return B


def is_converging(B):
    nb = np.linalg.norm(B, np.inf)

    if show_steps:
        print("Iteration auf Konvergenz prüfen. Iteration konvergiert, wenn ‖B‖ < 1. ‖B‖ = " + str(nb))

    return nb < 1


def iterate_jacobi(A, B, b, x0, tol):
    dim = b.shape[0]  # matrix size
    x = [x0]  # array of all step results x0, x1, ... xn
    err = 'N/A'

    while len(x) < 2 or err > tol:
        x_curr = np.zeros(dim)
        for i in range(dim):
            sum_ = 0
            for j in range(dim):
                if i == j: continue
                sum_ += A[i, j] * x[-1][j]

            x_curr[i] = (1 / A[i, i]) * (b[i] - sum_)

        x.append(x_curr)
        err = calculate_error_a_posteriori(B, x[-1], x[-2])

        if show_steps:
            print("n = " + str(len(x) - 1) + ": x" + str(len(x) - 1) + " = " + str(x_curr) + ", a-posteriori Fehler: " + str(err))

    return x


def calculate_error_a_posteriori(B, xn, xnm1):
    norm_b = np.linalg.norm(B, np.inf)

    err = (norm_b / (1 - norm_b)) * np.linalg.norm((xn - xnm1), np.inf)
    return err


def calculate_required_steps_a_priori(B, x0, x1, tol):
    norm_b = np.linalg.norm(B, np.inf)
    n_min = np.log((-(norm_b - 1) * tol) / (np.linalg.norm(x1 - x0, np.inf))) / np.log(norm_b)
    return n_min


xn, n, n2 = solve_jacobi(A, b, x0, tol)

print("\n==========================")
print('Lösung des Gleichungssystems durch Jacobi:')
print('x = ' + str(xn))
print('Benötigte Iterationen: n = ' + str(n))
print('Theoretisch min. benötigte Iterationen gem. a-priori-Abschätzung: n2 = ' + str(n2))
print('')
