import sympy as sy
from decimal import Decimal
import matplotlib.pyplot as plt
import math

"""Für Aufgaben der Art
'Das Polynom vierten Grades
f(x) = 230x^4 + 18x^3 + 9x^2 − 221x − 9
besitzt zwei reelle Nullstellen, die erste x1 im Intervall [−1, 0] und die zweite x2 im Intervall [0, 1].
a) Versuchen Sie, diese Nullstellen mit einer Fixpunktiteration xn+1 = F(xn) bis auf 10^−6 genau zu bestimmen.
Stellen Sie dafür die entsprechende Fixpunktgleichung F(x) = x auf und wählen Sie geeignete Startwerte gemäss
der Abbildung. Was stellen Sie bzgl. der Nullstelle in [0, 1] fest? Weshalb?

@version: 1.0, 23.01.2021
@author: zahlesev
"""

x = sy.symbols('x')

"""==================== INPUT ===================="""
# ACHTUNG: Für Funktionen wie cos usw die Sympy-Befehle brauchen, z.B. sy.cos(), sy.exp() usw.
F = (1.0/221.0) * (230.0 * x ** 4.0 + 18.0 * x ** 3.0 + 9.0 * x ** 2.0 - 9.0)  # Linke Seite der Fixpunktgleichung F(x) = x
x0 = 0  # Startwert der Iteration. (Setze x0 = 1 für divergierenden Fixpunkt in [0, 1])
precision = 1e-6  # Wie genau die Lösung sein soll. Bei 10^-6 = 1e-6
min_iterations = 5  # Minimale Anzahl Iterationen bei divergierenden Folgen, bevor abgebrochen wird.

# Ob die Fixpunktgleichung geplottet werden soll
show_plots = True
ap = -1; bp = 1  # Intervall, über welchem geplottet werden soll.
"""==============================================="""

# Plot F(x) and x
if show_plots:
    steps = 1000
    d = (Decimal(bp) - Decimal(ap)) / steps

    xvalues = []
    yvalues = []

    for i in range(steps + 1):
        xvalue = Decimal(ap) + i * d
        yvalue = F.subs(x, xvalue)

        xvalues.append(xvalue)
        yvalues.append(yvalue)

    plt.figure(1)
    plt.plot(xvalues, yvalues, label="F(x)")
    plt.plot(xvalues, xvalues, label="x")
    plt.xlim(ap, bp)
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.title("Fixpunktiteration F(x) = x")
    plt.legend()
    plt.show()

print("FIXPUNKTITERATION")
print("=================")
xn = [x0, F.subs(x, x0)]

print("Δ := Differenz zwischen den letzten zwei Resultaten.\n")

print("n = 0: x0 = " + str(xn[0]))
print("n = 1: x1 = F(x0) = " + str(xn[1]))

n = 1
while abs(xn[n] - xn[n-1]) > precision:
    n += 1

    xn.append(F.subs(x, xn[n-1]))

    if n > min_iterations and xn[n] > xn[n-1]:
        print("Folge divergiert! Kein Fixpunkt!")
        break

    print("n = " + str(n) + ": x" + str(n) + " = F(x" + str(n - 1) + ") = " + str(xn[n]) + ", Δ = " + str(abs(xn[n] - xn[n - 1])))

