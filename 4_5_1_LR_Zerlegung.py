# -*- coding: utf-8 -*-
"""
Lineares Gleichungssystem LGS Ax = b direkt mit LR-Zerlegung, mit oder ohne Spaltenpivotisierung, lösen.

@version: 1.0, 24.01.2021
@author: zahlesev@students.zhaw.ch
"""

import numpy as np
import math

"""==================== INPUT ===================="""
A = np.array([[0.8, 2.2, 3.6],
              [2.0, 3.0, 4.0],
              [1.2, 2.0, 5.8]], dtype=np.float64)
b = np.array([2.4, 1.0, 4.0], dtype=np.float64)

use_pivoting = True  # Ob mit Spaltenpivotisierung gearbeitet werden soll.

show_steps = True  # Ob die Zwischenresultate ausgegeben werden sollen
number_format = "{b:6.2f}"  # Formatierung der Zahl bei der Ausgabe von Matrizen
"""==============================================="""


def lr_solve(A, b):

    if A.shape[0] != A.shape[1]:
        raise Exception('A is not a quadratic matrix!')

    if A.shape[0] != b.shape[0]:
        raise Exception('b has not the same length as the dimensions of A.')

    # Copy input variables so that external data is not altered, and
    # to ensure that concurrent processes don't alter local data.
    R = np.copy(A)
    L = np.eye(R.shape[0])
    P = np.eye(R.shape[0])
    v = np.copy(b)

    if show_steps:
        print("1. Schritt: Bestimme R, P und L mit Gauss-Verfahren.")
        print_rlp(R, L, P)

    # Recursively create upper triangle matrix, and apply operations to b.
    calculate_r_l_p(R, L, P)

    Pb = P @ v
    if show_steps:
        print("\n2. Schritt: Löse Gleichungssystem Ly = Pb nach y durch Vorwärtseinsetzen auf.")
        print("b = " + str(v))
        print("Pb = " + str(Pb))

    # Solve Ly = Pb
    y = forward_solve_ly_eq_pb(L, Pb)

    if show_steps:
        print("\ny = " + str(y))
        print("\n3. Schritt: Löse Gleichungssystem Rx = y nach x durch Rückwärtseinsetzen auf.")

    x = reverse_solve_rx_eq_y(R, y)

    if show_steps:
        print("\nx = " + str(x))

    return R, L, P, x


def calculate_r_l_p(R, L, P):

    for i in range(R.shape[0]):

        # Apply pivoting if desired, or else ensure no 0 in first row.
        switch_rows_for_column_pivoting(R, L, P, i)

        # Create zeros in first column
        for row in range(i + 1, R.shape[0]):
            factor = -(R[row, i] / R[i, i])

            # Add first row times factor to current row
            R[row, :] = R[row, :] + (factor * R[i, :])

            # Store factor in L matrix
            L[row, i] = -factor

            if show_steps:
                print("Erzeuge 0 in Zeile " + str(row + 1) + " (Faktor = " + str(-factor) + "):")
                print_rlp(R, L, P)


def switch_rows_for_column_pivoting(R, L, P, i):
    pivot = 0
    pivot_row = i

    for row in range(i, R.shape[0]):
        if abs(R[row, i]) > pivot:
            pivot = abs(R[row, i])
            pivot_row = row

        # If no pivoting is desired, just take first row that's not zero.
        if not use_pivoting and pivot > 0:
            break

    if pivot == 0:
        raise Exception('All elements in first column are zero. System has no single solution.')

    if pivot_row > i:
        # Switch pivot row with first row

        # Switch rows in R-Matrix (all columns)
        r_copy = np.copy(R)
        R[i, :] = r_copy[pivot_row, :]
        R[pivot_row, :] = r_copy[i, :]

        # Switch rows in P-Matrix (all columns)
        p_copy = np.copy(P)
        P[i, :] = p_copy[pivot_row, :]
        P[pivot_row, :] = p_copy[i, :]

        # Switch rows in L-Matrix (only first i-1 columns)
        l_copy = np.copy(L)
        L[i, :i] = l_copy[pivot_row, :i]
        L[pivot_row, :i] = l_copy[i, :i]

        if show_steps:
            print("|Pivot| in Spalte " + str(i + 1) + " ist " + str(pivot) + " --> Vertausche Zeile 1 mit Zeile " + str(pivot_row + 1) + ".")
            print_rlp(R, L, P)

    else:
        if show_steps:
            print("Keine Zeilenvertauschungen für Spalte " + str(i + 1) + " notwendig!")

    return


def forward_solve_ly_eq_pb(L, Pb):
    size = L.shape[0]
    y = np.zeros(size)

    for row in range(size):
        y[row] = - (1 / L[row, row]) * (-Pb[row] + np.sum(L[row, :row] * y[:row]))

    return y


def reverse_solve_rx_eq_y(R, y):
    size = R.shape[0]
    x = np.zeros(size)

    # Start at last row and continue towards first row
    for row in range(size - 1, -1, -1):
        x[row] = - (1 / R[row, row]) * (-y[row] + np.sum(R[row, row + 1:] * x[row + 1:]))

    return x


def print_rlp(R, L, P):
    out = ""

    for row in range(R.shape[0]):
        print_matrix_names = int((R.shape[0] + 1) / 2) - 1 == row

        lbracket = "|"
        rbracket = "|"
        if row == 0: lbracket = "/"; rbracket = "\\";
        if row == R.shape[0] - 1: lbracket = "\\"; rbracket = "/"
        if R.shape[0] == 1: lbracket = "("; rbracket = ")"

        # Print R
        if print_matrix_names: out += "  R = "
        else: out += "..    "
        out += lbracket
        out += " "
        for col in range(R.shape[1]):
            out += number_format.format(b=R[row, col])
            out += " "
        out += rbracket

        # Print P
        if print_matrix_names: out += "  P = "
        else: out += "..    "
        out += lbracket
        out += " "
        for col in range(P.shape[1]):
            out += str(int(P[row, col]))
            out += " "
        out += rbracket

        # Print L
        if print_matrix_names: out += "  L = "
        else: out += "..    "
        out += lbracket
        out += " "
        for col in range(L.shape[1]):
            out += number_format.format(b=L[row, col])
            out += " "
        out += rbracket

        out += "\n"

    print(out)


"""
MAIN PROGRAM
"""
R, L, P, x = lr_solve(A, b)

print("\n\nObere Dreiecksmatrix R:")
print(R)
print("\nNormierte untere Dreiecksmatrix L:")
print(L)
print("\nPermutationsmatrix P:")
print(P)
print('\n\nLösung des Gleichungssystems ist: ')
for i in range(len(x)):
    print("x{a} = {b:5.2f}".format(a=i, b=x[i]), end='\t')
