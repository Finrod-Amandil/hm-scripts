"""
Für Aufgaben vom Typ 'Untersuchen Sie die Fehlerfortpflanzung im linearen Gleichungssystem Ax = b mit A = ... und b = ...
für den Fall, dass die rechte Seite von b in jeder Komponente um maximal 0.1 von b abweicht.

@author: zahlesev@students.zhaw.ch
@version: 1.0, 24.01.2021
"""

import numpy as np

"""==================== INPUT ===================="""
A = np.array([[2, 4], [4, 8.1]], dtype=np.float64)
b = np.array([1, 1.5], dtype=np.float64)

db = 0.1  # Maximaler absoluter Fehler in b
"""==============================================="""

print("Betrachte gestörtes Gleichungssystem Ax~ = b~, wobei b~ in jeder Komponente um maximal " + str(db) + " von b abweicht.")
print("Wähle für Berechnung die ∞-Norm, da dann gerade gilt ‖b~ - b‖∞ ≤ " + str(db))

niA = np.max([np.sum([abs(aij) for aij in A[i, :]]) for i in range(A.shape[0])])
print("Berechne Zeilensummennorm ‖A‖∞ = " + str(niA))

Ainv = np.linalg.inv(A)
print("Bestimme A⁻¹ = \n" + str(Ainv))

niAinv = np.max([np.sum([abs(aij) for aij in Ainv[i, :]]) for i in range(Ainv.shape[0])])
print("Berechne Zeilensummennorm ‖A⁻¹‖∞ = " + str(niAinv))

condA = niA * niAinv
print("Berechne cond(A) = ‖A‖∞ * ‖A⁻¹‖∞ = " + str(condA))

dxabs = niAinv * db

nib = np.max([abs(bi) for bi in b])
dxrel = condA * (db / nib)
print("")
print("Es gilt für den absoluten Fehler von x: ‖x - x~‖∞ ≤ ‖A⁻¹‖∞ * ‖b~ - b‖∞ = " + str(niAinv) + " * " + str(db) + " = " + str(dxabs))
print("Es gilt für den relativen Fehler von x: ‖x - x~‖∞ / ‖x‖∞ ≤ cond(A) * (‖b~ - b‖∞ / ‖b‖∞) = " + str(condA) + " * (" + str(db) + " / " + str(nib) + ") = " + str(dxrel))

print("")

print("Interpretation: Die Lösung x~ des gestörten Systems Ax~ = b~ wird von der Lösung x des exakten Systems Ax = b")
print("in jeder Komponente um maximal " + str(dxabs) + " abweichen (absoluter Fehler), und der relative Fehler wird")
print("maximal " + str(dxrel) + " betragen.")