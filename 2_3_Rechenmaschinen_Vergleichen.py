from decimal import Decimal

"""Für Aufgaben der Art
    'Gegeben seien zwei verschiedene Rechenmaschinen. Die erste davon arbeite mit einer 46-
    stelligen Binärarithmetik und die zweite einer 14-stelligen Dezimalarithmetik.
    Welche Maschine rechnet genauer? (Mit Begründung!)'
    
    @version: 1.0, 23.01.2021
    @author: zahlesev@students.zhaw.ch
"""

"""==================== INPUT ===================="""
b1 = 8   # Basis der ersten Rechenmaschine (Binär = 2, Dezimal = 10, Oktal = 8, Hexadezimal = 16)
n1 = 2  # Anzahl Mantisse-Stellen der ersten Rechenmaschine

b2 = 2  # Basis der zweiten Rechenmaschine (Binär = 2, Dezimal = 10, Oktal = 8, Hexadezimal = 16)
n2 = 4  # Anzahl Mantisse-Stellen der zweiten Rechenmaschine
"""==============================================="""

print("Genauigkeit einer Rechenmaschine eps = B/2 * B^(-n) bei Basis B und n Mantisse-Stellen.\n")

eps1 = (Decimal(b1) / Decimal(2)) * (Decimal(b1)**Decimal(-n1))
print("eps1 = " + str(b1) + "/2 * " + str(b1) + "^-" + str(n1) + " = " + str(eps1))

eps2 = (Decimal(b2) / Decimal(2)) * (Decimal(b2)**Decimal(-n2))
print("eps2 = " + str(b2) + "/2 * " + str(b2) + "^-" + str(n2) + " = " + str(eps2))

print("\nDiejenige Maschine mit kleinerem Epsilon (eps) rechnet genauer.")

if eps1 < eps2:
    print("Da eps1 < eps2 ist, rechnet Maschine 1 genauer.")
elif eps2 < eps1:
    print("Da eps2 < eps1 ist, rechnet Maschine 2 genauer.")
else:
    print("Beide Maschinen rechnen gleich genau, da die Epsilon gleich gross sind.")
