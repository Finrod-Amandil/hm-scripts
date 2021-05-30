import sympy as sy
import math

"""
@version: 1.0, 23.01.2021
@author: zahlesev@students.zhaw.ch
"""

x = sy.symbols('x')

"""==================== INPUT ===================="""
f = sy.exp(x)-3-sy.sqrt(x)
x0 = 1.75
max_error = 1e-6
"""==============================================="""

df = sy.diff(f, x)
d2f = sy.diff(df, x)
fl = sy.lambdify(x, f)
dfl = sy.lambdify(x, df)
d2fl = sy.lambdify(x, d2f)

print("f'(x) = " + str(df))

print("Konvergenzbedingung für x0 prüfen:")
d = abs((fl(x0) * d2fl(x0)) / ((dfl(x0)) ** 2))

if d < 1:
    print("Konvergenzbedingung erfüllt!")
else:
    print("Konvergenzbedingung NICHT erfüllt!")

if d < 1:
    xn = [x0]
    print("n = 0: x0 = " + str(x0))

    n = 0

    while n < 1 or abs(xn[n] - xn[n - 1]) > max_error:
        xn.append(xn[n] - fl(xn[n]) / dfl(xn[n]))

        n += 1

        print("n = " + str(n) + ": x" + str(n) + " = " + str(xn[n]) + ", Δ = " + str(abs(xn[n] - xn[n - 1])))
