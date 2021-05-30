import sympy as sy

x, y, z = sy.symbols('x y z')

f = x**3 + x * y + sy.sin(y)  # ACHTUNG: Für sinus/cosinus/Exponentialfunktion immer sy.sin/sy.cos/sy.exp verwenden!
print('f = ' + str(f))

# 1. Partielle Ableitung nach x
dfx1 = sy.diff(f, x)
print('dfx1 = ' + str(dfx1))

# 1. Partielle Ableitung nach y
dfy1 = sy.diff(f, y)
print('dfy1 = ' + str(dfy1))

# 2. Partielle Ableitung nach x
dfx2 = sy.diff(f, x, x)
print('dfx2 = ' + str(dfx2))

# 2. Partielle Ableitung nach y
dfy2 = sy.diff(f, y, y)
print('dfy2 = ' + str(dfy2))

# Partielle Ableitung nach x und y
dfxy = sy.diff(f, x, y)
print('dfxy = ' + str(dfxy))
