# -*- coding: utf-8 -*-
"""
Lineares Gleichungssystem LGS Ax = b direkt mit Gauss-Algorithmus lösen und Determinante von A berechnen

OHNE SPALTENPIVOTISIERUNG, je nach Input mit Präzision 32 Bit oder 64 Bit Gleitkommazahl

@author: IT19ta_WIN / zahlesev@students.zhaw.ch
@version: 1.1, 25.01.2021
"""


import numpy as np

"""==================== INPUT ===================="""
A = np.array([[10, 6], [-20, -12]], dtype=np.float64)
b = np.array([0, 0], dtype=np.float64)

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

    # Ensure that there is no zero in top left cell
    row_inversion_count += ensure_non_zero_leading_element(R, v)

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


def ensure_non_zero_leading_element(R, v):
    non_zero_row_found = False
    row_inversion_count = 0

    if R[0, 0] != 0:
        if show_steps:
            print("Keine Zeilenvertauschungen notwendig!")
        return row_inversion_count

    for row in range(R.shape[0]):
        if R[row, 0] == 0:
            continue

        # Switch non-zero row with first row
        r_copy = np.copy(R)
        R[0, :] = r_copy[row, :]
        R[row, :] = r_copy[0, :]

        v_copy = np.copy(v)
        v[0] = v_copy[row]
        v[row] = v_copy[0]

        if show_steps:
            print("Vertausche Zeile 1 mit Zeile " + str(row + 1) + ".")
            print_extended_matrix(R, v)

        non_zero_row_found = True
        row_inversion_count += 1
        break

    if not non_zero_row_found:
        raise Exception('All elements in first column are zero. System has no single solution.')

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
