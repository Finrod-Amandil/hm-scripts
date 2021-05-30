"""
Berechnet die 1-Norm, 2-Norm und ∞-Norm für Vektoren und Matrizen

@author: zahlesev@students.zhaw.ch
@version: 1.0, 24.01.2021
"""

import numpy as np
import math

"""==================== INPUT ===================="""
A = np.array([[1, 2, 3], [3, 4, -2], [7, -3, 5]], dtype=np.float64)
b = np.array([-1, 2, 3], dtype=np.float64)
"""==============================================="""

n1b = np.sum([abs(bi) for bi in b])
print("Summennorm ‖b‖₁ = " + str(n1b))

n2b = math.sqrt(np.sum([bi ** 2 for bi in b]))
print("Euklidische Norm ‖b‖₂ = " + str(n2b))

nib = np.max([abs(bi) for bi in b])
print("Maximumnorm ‖b‖∞ = " + str(nib))

print("")

n1A = np.max([np.sum([abs(aij) for aij in A.T[j, :]]) for j in range(A.shape[1])])
print("Spaltensummennorm ‖A‖₁ = " + str(n1A))

niA = np.max([np.sum([abs(aij) for aij in A[i, :]]) for i in range(A.shape[0])])
print("Zeilensummennorm ‖A‖∞ = " + str(niA))

print("")

Ainv = np.linalg.inv(A)
print("A⁻¹ = \n" + str(Ainv))

niAinv = np.max([np.sum([abs(aij) for aij in Ainv[i, :]]) for i in range(Ainv.shape[0])])
condAi = niA * niAinv
print("‖A⁻¹‖∞ = " + str(niAinv))
print("Kondition bez. ∞-Norm: cond(A)∞ = ‖A‖∞ * ‖A⁻¹‖∞ = " + str(condAi))

n1Ainv = np.max([np.sum([abs(aij) for aij in Ainv.T[j, :]]) for j in range(Ainv.shape[1])])
condA1 = n1A * n1Ainv
print("‖A⁻¹‖₁ = " + str(n1Ainv))
print("Kondition bez. 1-Norm: cond(A)₁ = ‖A‖₁ * ‖A⁻¹‖₁ = " + str(condA1))
