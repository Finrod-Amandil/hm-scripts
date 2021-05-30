# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 13:21:19 2020

Lineares Gleichungssystem LGS Ax = b direkt mit Gauss-Algorithmus lösen und Determinante von A berechnen

MIT SPALTENPIVOTISIERUNG

@author: IT19ta_WIN / zahlesev@students.zhaw.ch
"""


import numpy as np

"""==================== INPUT ===================="""
A = np.array([[   4,  -1,  -5],
               [-12,   4,  17],
               [ 32, -10, -41]],
              dtype=np.float64)
b = np.array([-5, 19, -39], dtype=np.float64)

A = np.array([[1e-5, 1e-5], [2, 3]], dtype=np.float64)
b = np.array([1e-5, 1], dtype=np.float64)

show_steps = True  # Ob die Zwischenresultate ausgegeben werden sollen
number_format = "{b:8.2f}"  # Formatierung der Zahl bei der Ausgabe von Matrizen
"""==============================================="""


def gauss_solve(A, b):

    if A.shape[0] != A.shape[1]:
        raise Exception('A is not a quadratic matrix!')

    if A.shape[0] != b.shape[0]:
        raise Exception('b has not the same length as the dimensions of A.')

    # Copy input variables so that external data is not altered, and
    # to ensure that concurrent processes don't alter local data.
    R = np.copy(A)
    v = np.copy(b)

    # Recursively create upper triangle matrix, and apply operations to b.
    row_inversion_count = create_upper_triangle_matrix(R, v, 0)

    # Determine solution by reverse-solving with triangle matrix.
    x = reverse_solve(R, v)

    # Calculate determinant as product of diagonal elements
    det = (-1) ** row_inversion_count
    for row in range(R.shape[0]):
        det *= R[row, row]

    return R, det, x


def create_upper_triangle_matrix(R, v, row_inversion_count):
    if show_steps:
        print("Betrachte jetzt Sub-Matrix der Grösse " + str(len(R)) + ":")
        print_extended_matrix(R, v)

    if R.shape[0] == 1 and R.shape[1] == 1:
        return 0

    # If largest value in column is not in first row, switch rows to improve condition / reduce error
    row_inversion_count += switch_rows_for_column_pivoting(R, v)

    # Create zeros in first column
    for row in range(1, R.shape[0]):
        factor = -(R[row, 0] / R[0, 0])

        # Add first row times factor to current row
        R[row, :] = R[row, :] + (factor * R[0, :])
        v[row] = v[row] + (factor * v[0])

        if show_steps:
            print("Erzeuge 0 in Zeile " + str(row + 1) + ":")
            print_extended_matrix(R, v)

    # Repeat process on sub-matrix
    row_inversion_count = create_upper_triangle_matrix(R[1:, 1:], v[1:], row_inversion_count)

    return row_inversion_count


def switch_rows_for_column_pivoting(R, v):
    row_inversion_count = 0

    pivot = 0
    pivot_row = 0

    for row in range(R.shape[0]):
        if abs(R[row, 0]) > pivot:
            pivot = abs(R[row, 0])
            pivot_row = row

    if pivot == 0:
        raise Exception('All elements in first column are zero. System has no single solution.')

    if pivot_row > 0:
        # Switch pivot row with first row
        r_copy = np.copy(R)
        R[0, :] = r_copy[pivot_row, :]
        R[pivot_row, :] = r_copy[0, :]

        v_copy = np.copy(v)
        v[0] = v_copy[pivot_row]
        v[pivot_row] = v_copy[0]

        row_inversion_count += 1
        if show_steps:
            print("|Pivot| ist " + str(pivot) + " --> Vertausche Zeile 1 mit Zeile " + str(pivot_row + 1) + ".")
            print_extended_matrix(R, v)

    else:
        if show_steps:
            print("Keine Zeilenvertauschungen notwendig!")

    return row_inversion_count


def reverse_solve(R, v):
    size = R.shape[0]
    x = np.zeros(size)

    # Start at last row and continue towards first row
    for row in range(size - 1, -1, -1):
        x[row] = - (1 / R[row, row]) * (-v[row] + np.sum(R[row, row + 1:] * x[row + 1:]))

    return x


def print_extended_matrix(R, v):
    out = ""

    for row in range(R.shape[0]):
        lbracket = "|"
        rbracket = "|"

        if row == 0: lbracket = "/"; rbracket = "\\";
        if row == R.shape[0] - 1: lbracket = "\\"; rbracket = "/"
        if R.shape[0] == 1: lbracket = "("; rbracket = ")"

        out += lbracket
        out += " "

        for col in range(R.shape[1]):
            out += number_format.format(b=R[row, col])
            out += " "

        out += "| "
        out += number_format.format(b=v[row])
        out += " "
        out += rbracket
        out += "\n"

    print(out)


"""
MAIN PROGRAM
"""
R, det, x = gauss_solve(A, b)

print("\nDeterminante von A:", str(det))
print("\nObere Dreiecksmatrix:")
print(R)
print('\nLösung des Gleichungssystems ist: ')
for i in range(len(x)):
    print("x{a} = {b:5.2f}".format(a=i, b=x[i]), end='\t')

print("\n\nLösung mit numpy.linalg.solve: x = " + str(np.linalg.solve(A, b)))
