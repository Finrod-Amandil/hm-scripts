"""
Iterative Berechnung der Maschinengenauigkeit eps für binäre Basis

@author: zahlesev@students.zhaw.ch
"""

eqmin = 0
x = 1.0

while 1 + x != 1:
    x = x * 0.5
    eqmin -= 1

"""Exponent of eps = e + 1"""
print('e = {}'.format(eqmin))


eqmax = 0
x = 1.0

while x + 1 != x:
    x = x * 2.0
    eqmax += 1

"""Exponent of qmax = e - 1"""
print('e = {}'.format(eqmax))
