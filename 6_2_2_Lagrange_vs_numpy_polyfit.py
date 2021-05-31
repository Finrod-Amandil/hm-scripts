import numpy as np
import matplotlib.pyplot as plt

x_in = np.array([1981, 1984, 1989, 1993, 1997, 2000, 2001, 2003, 2004, 2010])
y_in = np.array([0.5, 8.2, 15, 22.9, 36.6, 51, 56.3, 61.8, 65, 76.7], dtype=np.float64)

coeff = np.polyfit(x_in, y_in, x_in.shape[0] - 1)

x = np.arange(1975, 2020, 0.1)

plt.figure(1)
plt.grid()
plt.plot(x, np.polyval(coeff, x), zorder=0, label='polyfit')
plt.scatter(x_in, y_in, marker='x', color='r', zorder=1, label='measured')
plt.scatter(2020, np.polyval(coeff, 2020), marker='x', color='fuchsia', zorder=1, label='extrapolated', lw=3)
plt.legend()
plt.show()


""" 
b) Was ist der Schätzwert für 2020, basierend auf Ihrem Resultat aus a)? Ist das realistisch, und können solche
   Polynome hoher Ordnung für Schätzwerte ausserhalb des Intervalls der vorhandenen Datenwerte benutzt
   werden? 

A: Der interpolierte Wert beträgt ca. 125, was bedeuten würde, dass 125% aller Haushalte einen PC besitzen
   Dies ist offensichtlich keine sinnvolle Angabe. Diese Methode ist wohl nur für die Interpolation, aber nicht
   für die Extrapolation von fehlenden Werten sinnvoll.

"""


def lagrange_int(x, y, x_int):
    return [np.sum([y[i] * np.prod([(x_int_i - x[j] * 1.0) / (x[i] - x[j]) for j in range(x.shape[0]) if j != i]) for i in range(x.shape[0])]) for x_int_i in x_int]


x = np.arange(1981, 2010, 0.1)

plt.figure(2)
plt.grid()
plt.plot(x, lagrange_int(x_in, y_in, x), zorder=0, label="lagrange")
plt.plot(x, np.polyval(coeff, x), zorder=1, label='polyfit')
plt.scatter(x_in, y_in, marker='x', color='r', zorder=2, label='measured')
plt.ylim([-100, 250])
plt.legend()
plt.show()

"""
c) Was stellen Sie im Vergleich der beiden Methoden fest? 

A: Das Lagrange-Polynom oszilliert im Bereich der äusseren Messwerte stark und bietet dort auch bei der Interpolation
   keine guten Näherungswerte.
"""