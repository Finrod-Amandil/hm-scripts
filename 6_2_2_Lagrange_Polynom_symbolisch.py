import sympy as sy

x = sy.symbols('x')

x_val = sy.Matrix([0, 2500, 5000, 10000])  # xi
y_val = sy.Matrix([1013, 747, 540, 226])   # yi
x_int = 3750  # Zu interpolierende Stelle

n = x_val.shape[0] - 1

p = 0

for i in range(n + 1):

    l = 1
    for j in range(n + 1):
        if j == i:
            continue

        l *= (x - x_val[j])/(x_val[i] - x_val[j])

    p += y_val[i] * l

print('Pn(x) = ' + str(sy.simplify(p)))
print('\nLATEX (Zum Anschauen eingeben unter https://www.codecogs.com/latex/eqneditor.php):')
print('P_n(x)=' + str(sy.latex(sy.simplify(p))))
print()

p = p.subs(x, x_int)
y_int = p.evalf()

print('Pn({}) = {}'.format(x_int, y_int))
