import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4], dtype=np.float64)  # xi
y = np.array([6, 6.8, 10, 10.5], dtype=np.float64)  # yi

print('Ansatz für lineare Regression: f(x) = ax + b = a * f1(x) + b * f2(x) mit f1(x) = x, f2(x) = 1')
print('Minimiere das Fehlerfunktional E(f)(a, b) = ∑[i = 1 .. n](yi - (a*xi + b))^2   (Quadrierte Differenz zwischen den Messwerten yi und den Schätzwerten von f(x))')
print('Die partiellen Ableitungen des Fehlerfunktionals nach a und nach b liefern zwei Gleichungen, als LGS Ax = r')
print('⎡ ∑xi^2   ∑xi ⎤   ⎡ a ⎤   ⎡ ∑xi*yi ⎤\n' +
      '⎢              ⎥ * ⎢   ⎥ = ⎢        ⎥\n' +
      '⎣ ∑xi     n   ⎦    ⎣ b ⎦   ⎣ ∑yi    ⎦\n')

A = np.array([
    [np.sum(x ** 2), np.sum(x)],
    [np.sum(x), x.shape[0]]
])

r = np.array([np.sum(x * y), np.sum(y)])

print('A = \n{}'.format(A))
print('r = {}'.format(r))

print('LGS wird gelöst...\n')

ab = np.linalg.solve(A, r)
a = ab[0]
b = ab[1]

print('a = {}, b = {}'.format(a, b))
print('Die gesuchte Ausgleichsgerade ist also f(x) = {}x + {}'.format(a, b))


xx = np.arange(x[0], x[-1], (x[-1] - x[0]) / 10000)  # Plot-X-Werte
yy = a * xx + b

plt.figure(1)
plt.grid()
plt.plot(xx, yy, zorder=0)
plt.scatter(x, y, marker='x', color='r', zorder=1)
plt.show()
