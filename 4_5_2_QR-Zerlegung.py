# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 13:26:09 2020

Höhere Mathematik 1, Serie 8, Gerüst für Aufgabe 2

Description: calculates the QR factorization of A so that A = QR
Input Parameters: A: array, n*n matrix
Output Parameters: Q : n*n orthogonal matrix
                   R : n*n upper right triangular matrix
Remarks: none

@author: knaa, zahlesev@students.zhaw.ch
"""

import numpy as np

"""==================== INPUT ===================="""
A = np.array([[1, -2, 3], [-5, 4, 1], [2, -1, 3]], dtype=np.float64)
b = np.array([1, 9, 5], dtype=np.float64)

show_steps = True  # Ob die Zwischenresultate ausgegeben werden sollen
"""==============================================="""


def qr(A):

    A = np.copy(A)  # necessary to prevent changes in the original matrix A_in
    A = A.astype('float64')  # change to float

    n = np.shape(A)[0]

    if n != np.shape(A)[1]:
        raise Exception('Matrix is not square')

    Q = np.eye(n)
    R = A

    if show_steps: print("A = \n" + str(A) + "\nQ = \n" + str(Q))

    for j in np.arange(0, n - 1):
        if show_steps: print("\n\nIteration " + str(j + 1) + "\n--------------------------------")

        a = np.copy(R[j:, j]).reshape(n - j, 1)
        e = np.eye(n - j)[:, 0].reshape(n - j, 1)

        length_a = np.linalg.norm(a)
        if a[0] >= 0:
            sig = 1
        else:
            sig = -1
        v = a + sig * length_a * e

        if show_steps:
            print("\na" + str(j + 1) + " = \n" + str(a))
            print("e" + str(j + 1) + " = \n" + str(e))
            print("\nv" + str(j + 1) + " = a" + str(j + 1) + " + sign(a" + str(j + 1) + str(j + 1) + ") * |a" + str(j + 1) + "| * e" + str(j + 1))
            print("sign(a" + str(j + 1) + str(j + 1) + ") = " + str(sig))
            print("|a" + str(j + 1) + "| = " + str(length_a))
            print("=> v = \n" + str(v))

        u = v / (np.linalg.norm(v))

        if show_steps:
            print("\nu" + str(j + 1) + " = 1 / |v" + str(j + 1) + "| * v" + str(j + 1))
            print("=> u" + str(j + 1) + " = \n" + str(u))

        H = np.eye(n - j) - (2 * u * u.T)

        Qi = np.eye(n)
        Qi[j:, j:] = H

        if show_steps:
            print("\nH" + str(j + 1) + " = Iₙ - 2 * u" + str(j + 1) + " * u" + str(j + 1) + "ᵀ")
            print("=> H" + str(j + 1) + " = \n" + str(H))
            print("=> Q" + str(j + 1) + " = \n" + str(Qi))

        R = np.matmul(Qi, R)
        Q = np.matmul(Q, Qi.T)

        if show_steps:
            print("\nR = Q" + str(j + 1) + " * R")
            print("=> R = \n" + str(R))
            print("\nQ = Q * Q" + str(j + 1) + "ᵀ")
            print("=> Q = \n" + str(Q))

    print("=================================")
    return Q, R


def reverse_solve(R, b):
    size = R.shape[0]
    x = np.zeros(size)

    # Start at last row and continue towards first row
    for row in range(size - 1, -1, -1):
        x[row] = - (1 / R[row, row]) * (-b[row] + np.sum(R[row, row + 1:] * x[row + 1:]))

    return x


Q, R = qr(A)

print('A = \n' + str(A))
print('\nQ = \n' + str(Q))
print('\nR = \n' + str(R))

QTb = Q.T @ b
x = reverse_solve(R, QTb)
if show_steps:
    print("\nLösung der Gleichung Rx = Qᵀb durch Rückwärtseinsetzen bestimmen.")
    print("Qᵀb = " + str(QTb))

print("\nx = " + str(x))
