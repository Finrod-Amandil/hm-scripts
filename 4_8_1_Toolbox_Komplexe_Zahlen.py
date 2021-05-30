"""
    Toolbox fürs Rechnen mit Komplexen Zahlen mit numpy

    @version: 1.0, 24.01.2021
    @author: zahlesev@students.zhaw.ch
"""

import matplotlib.pyplot as plt
import numpy as np
import math

# Komplexe Zahl definieren:
z = 2 + 3j
z = complex(2, 3)


# Real-, Imaginärteil, Komplex-Konjugierte, Betrag / Länge, Winkel
print("z = " + str(z))
print("Realteil von z: Re(z) = " + str(z.real))
print("Imaginärteil von z: Im(z) = " + str(z.imag))
print("Komplex-konjugierte von z: z* = " + str(z.conjugate()))
print("Länge / Radius von z: r = " + str(np.sqrt(z * z.conjugate()).real))
print("Winkel von z in Radian: φ = " + str(np.angle(z)))
print("Winkel von z in Grad: φ = " + str(np.angle(z, deg=True)))
print("Winkel von z als Vielfaches von π: φ = " + str(np.angle(z) / math.pi) + "π")


# Darstellungsformen
z = 3 + 4j  # Normalform
z = 2 * (math.cos(2 / 3 * math.pi) + math.cos(2 / 3 * math.pi) * 1j)  # Trigonometrische Form (Winkel in Radian)
z = 2 * (math.cos(120 * math.pi / 180) + math.cos(120 * math.pi / 180) * 1j)  # Trigonometrische Form (Winkel in Grad)
z = 2 * np.exp(2 / 3 * math.pi * 1j)  # Exponentialform / Polarform

print("z = " + str(z))  # Normalform

r = np.sqrt(z * z.conjugate()).real
phi = np.angle(z)
phi_deg = np.angle(z, deg=True)

print("z = {r} * (cos({phi}) + j*sin({phi})))".format(r=np.round(r, 2), phi=np.round(phi, 2)))  # Trigonometrische Form (Winkel in Radian)
print("z = {r} * (cos({phi_deg}°) + j*sin({phi_deg}°)))".format(r=np.round(r, 2), phi_deg=np.round(phi_deg, 2)))  # Trigonometrische Form (Winkel in Radian)

print("z = {r} * e^{phi}j".format(r=np.round(r, 2), phi=np.round(phi, 2)))  # Exponentialform


# Grundoperationen
z1 = 3 + 4j
z2 = -7 - 1j

print("z1 = " + str(z1))
print("z2 = " + str(z2))
print("Summe: z1 + z2 = " + str(z1 + z2))
print("Differenz: z1 - z2 = " + str(z1 - z2))
print("Produkt: z1 * z2 = " + str(z1 * z2))
print("Verhältnis: z1 / z2 = " + str(z1 / z2))
print("Potenz: zi ^ z2 = " + str(z1 ** z2))


# Nullstellen eines Polynoms
print("P(z) = z^3 - z^2 + 4z - 4")
print("Löse P(z) = 0")
x = np.roots([1, -1, 4, -4])  # Koeffizienten des Polynoms
x = np.round(x, 2)
print("=> x = " + str(x))


# Betrag eines komplexen Vektors
z = [1j, 3 + 3j, 4 - 7j]
absZ = np.sqrt(np.sum([zi * zi.conjugate() for zi in z])).real
print("Länge von z = " + str(z) + ": |z| = " + str(absZ))


# Komplexe Zahlen in der Gausschen Zahlenebene darstellen
cnums = [3 + 4j, 1j, 0, 7, -2 - 5j]
X = [x.real for x in cnums]
Y = [x.imag for x in cnums]
plt.grid()
plt.scatter(X, Y, color='red')
plt.show()




