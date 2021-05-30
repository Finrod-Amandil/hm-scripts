# -*- coding: utf-8 -*-
"""
Serie 1, Aufgabe 3

@author: IT19ta_WIN10 / zahlesev@students.zhaw.ch
"""

import numpy as np
import timeit

def fact_rec(n):
    # y = fact_rec(n) berechnet die Fakultät von n als fact_rec(n) = n * fact_rec(n -1) mit fact_rec(0) = 1
    # Fehler, falls n < 0 oder nicht ganzzahlig
    if n < 0 or np.trunc(n) != n:
        raise Exception('The factorial is defined only for positive integers')
        
    if n <= 1:
        return 1
    else:
        return n * fact_rec(n - 1)
    
def fact_for(n):
    # Fehler, falls n < 0 oder nicht ganzzahlig
    if n < 0 or np.trunc(n) != n:
        raise Exception('The factorial is defined only for positive integers')
        
    factorial = 1;
        
    for factor in range(1, n + 1):
        factorial = factor * factorial
        
    return factorial


t_rec = timeit.repeat("fact_rec(500)", "from __main__ import fact_rec", number=10)
t_for = timeit.repeat("fact_for(500)", "from __main__ import fact_for", number=10)

print(t_rec)
print(t_for)

print("Average factor of calculation time between recursive and iterative approach: ")
print(np.average(t_rec) / np.average(t_for))

print([str(n) + "! = " + str(fact_for(n)) for n in range(190, 201)])
print("float(170!) = " + str(float(fact_for(170))))
print("float(171!) = " + str(float(fact_for(171))))

"""
Q: Welche der beiden Funktionen ist schneller und um was für einen Faktor? Weshalb?

A: Die Zeitkomplexitäten beider Lösungen sind identisch, und zwar O(n) (mit der
   vereinfachenden Annahme das die Zeitkomplexität einer einzelnen Multiplikation
   gleich O(1) ist.)

   Dennoch ist die rekursive Lösung um den Faktor 9-10 langsamer als die iterative.
   Ein Grund dafür mag sein, dass durch die vielen Funktionsaufrufe mehr Variablen
   im Stack festgehalten werden müssen, was zusätzliche Zeit und Speicher-
   ressourcen erfordert.
   
Q: Gibt es in Python eine obere Grenze für die Fakultät von n
    - als ganze Zahl (vom Typ 'integer')? Versuchen Sie hierzu, das Resultat für n ∈ [190, 200] als 
      integer auszugeben.
    - als reelle Zahl (vom Typ 'float')? Versuchen Sie hierzu, das Resultat für n ∈ [170, 171] als 
      float auszugeben.

A: Es scheint, als wäre die Grösse / Länge eines Integers in Python ziemlich unlimitiert.
   Selbst 500! kann noch berechnet und ausgegeben werden. Uns fiel auf, dass es nicht funktioniert,
   wenn statt dem Python command 'range(1, n + 1)' das numpy-Äquivalent 'np.arange(1, n + 1)'
   gearbeitet wird, da dann statt mit dem offenbar unlimitierten "Python-Standard Integer"
   mit int32 gerechnet wird, welcher natürlich auf 32 Bit beschränkt ist, und für die meisten
   Fakultätwerte nicht ausreicht.
   
   Fakultäten grösser als 170! können jedoch nicht mehr als Gleitkommazahlen dargestellt werden.
   Dies deckt sich mit dem im Unterricht vermittelten Wissen über Gleitkommazahlen, und dass die
   Anzahl zur Verfügung stehender Bits für den Exponenten limitiert sind, womit die grösst-
   mögliche Zahl klar limitiert ist.

"""
