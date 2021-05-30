import sympy as sy

x1, x2, x3, x4, x5, x6, x7, x8, x9 = sy.symbols('x1, x2, x3, x4, x5, x6, x7, x8, x9')

"""
=======================================================================================================================
INPUT
=======================================================================================================================
"""

# ACHTUNG: Für sinus/cosinus/Exponentialfunktion immer sy.sin/sy.cos/sy.exp/sy.ln/sy.abs verwenden!
f = sy.Matrix([
    x1 + x2**2 - x3**2 - 13,
    sy.ln(x2 / 4) + sy.exp((x3 / 2) - 1) - 1,
    (x2 - 3)**2 - x3**3 + 7
])

x = sy.Matrix([x1, x2, x3])  # Wenn mehr oder weniger als 3 Variablen auftreten, diese Liste anpassen!
x0 = sy.Matrix([3/2, 3, 5/2])    # Stelle, an welcher die Jacobi-Matrix ausgewertet werden soll, z.B. x0 bei Newton

"""
=======================================================================================================================
"""

print('Bilde die Jacobi-Matrix Df(x) für f(x) = ' + str(f) + ' mit x = ' + str(x) + '.\n')

Df = f.jacobian(x)

print('Ganze Jacobi-Matrix: Df = ' + str(Df))
print('LATEX (Zum Anschauen eingeben unter https://www.codecogs.com/latex/eqneditor.php):')
print(sy.latex(Df))
print()

Dfx0 = Df
fx0 = f

# Ersetze alle xi-Variablen mit konkreten Werten
for i in range(x.shape[0]):
    Dfx0 = Dfx0.subs(x[i], x0[i])
    fx0 = fx0.subs(x[i], x0[i])

Dfx0_eval = Dfx0.evalf()
fx0_eval = fx0.evalf()

print('Jacobi-Matrix ausgewertet an Stelle x0 = ' + str(x0) + ': Df(x0) = ' + str(Dfx0_eval))
print('Funktion f ausgewertet an Stelle x0 = ' + str(x0) + ': f(x0) = ' + str(fx0_eval))
print()

g = fx0_eval + Dfx0_eval * (x - x0)

print('Linearisierung g(x) = f(x0) + Df(x0) * (x - x0) = ' + str(g))
print('LATEX (Zum Anschauen eingeben unter https://www.codecogs.com/latex/eqneditor.php):')
print(sy.latex(g))
