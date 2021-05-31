import numpy as np
import matplotlib.pyplot as plt

x = np.array([4, 6, 8, 10], dtype=np.float64)  # Stützpunkte (Knoten) xi
y = np.array([6, 3, 9,  0], dtype=np.float64)  # Stützpunkte (Knoten) yi
x_int = 9  # Zu interpolierender Wert

n = x.shape[0] - 1  # Anzahl Spline-Polynome
print('n = ' + str(n))

print('Die Natürlichen kubischen Spline-Polynome S0(x) .. S{}(x) haben die Form:'.format(n - 1))
print('Si(x) = ai + bi(x - xi) + ci(x - xi)^2 + di(x - xi)^3')
print('=> Bestimme die Koeffizienten a0..a2, b0..b2, c0..c2, d0..d2')

print('\n1. ai = yi')

a = [y[i] for i in range(n)]

for i in range(n):
    print('\ta{} = {}'.format(i, a[i]))

print('\n2. Hilfsgrössen hi = x(i+1) - xi (Distanz zwischen Stützstellen)')

h = [x[i + 1] - x[i] for i in range(n)]

for i in range(n):
    print('\th{} = {}'.format(i, h[i]))

print('\n3. c0 = 0, cn = 0')
c = [0 for i in range(n + 1)]

print('\n4. Berechne c1 .. c(n-1) aus dem LGS Ac = z')
print('\tA hat auf der Diagonalen die Elemente 2*(h(i-1) + hi und hi unter und über der Diagonalen (vgl. Skript S. 120)')
print('\tc = [c1 c2 .. c(n-1)]')
print('\tz = [3 * (y(i+1) - yi) / (hi) - 3 * (yi - y(i-1)) / (h(i-1)), usw.] vgl. Skript S. 120')

# Linear system for finding c1, ... cn-1
A = np.full((n - 1, n - 1), 0)
for i in range(n - 1):
    # Diagonal element
    A[i, i] = 2 * (h[i] + h[i + 1])

    if i < n - 2:
        # Element below diagonal
        A[i + 1, i] = h[i + 1]

        # Element right of diagonal
        A[i, i + 1] = h[i + 1]

print('\tA = \n' + str(A))

z = [3 * (y[i + 2] - y[i + 1]) / h[i + 1] - 3 * (y[i + 1] - y[i]) / h[i] for i in range(n - 1)]

print('\tz = \n' + str(z))
print('\tLöse damit Ac = z')

c[1:-1] = np.linalg.solve(A, z)

for i in range(n):
    print('\tc{} = {}'.format(i, c[i]))

print('\n5. bi = (y(i+1) - yi) / hi - (hi / 3) * (c(i + 1) + 2 * ci)')

b = [(y[i + 1] - y[i]) / (h[i]) - (h[i] / 3) * (c[i + 1] + 2 * c[i]) for i in range(n)]

for i in range(n):
    print('\tb{} = {}'.format(i, b[i]))

print('\n6. di = (1 / (3 * hi)) * (c(i+1) - ci)')

d = [(1 / (3 * h[i])) * (c[i + 1] - c[i]) for i in range(n)]

for i in range(n):
    print('\td{} = {}'.format(i, d[i]))

print('\nDiese Werte jetzt einsetzen in Si(x) = ai + bi(x - xi) + ci(x - xi)^2 + di(x - xi)^3:')

for i in range(n):
    print('\tS{}(x) = {} + {} * (x - {}) + {} * (x - {})^2 + {} * (x - {})^3'.format(i, a[i], b[i], x[i], c[i], x[i], d[i], x[i]))

print('\nBestimmen, welches Spline-Polynom verwendet werden muss (Vergleich mit den Stützstellen)')

i = np.max(np.where(x <= x_int))  # Finde die Stützstelle, deren x-Wert am grössten, aber gerade noch kleiner ist als x_int
print('Für x_int = {} muss S{} verwendet werden.'.format(x_int, i))

y_int = a[i] + b[i] * (x_int - x[i]) + c[i] * (x_int - x[i]) ** 2 + d[i] * (x_int - x[i]) ** 3

print('S{}({}) = {}'.format(i, x_int, y_int))



# PLOTTING
xx = np.arange(x[0], x[-1], (x[-1] - x[0]) / 10000)  # Plot-X-Werte

# Bestimme für jeden x-Wert, welches Spline-Polynom gilt
xxi = [np.max(np.where(x <= xxk)) for xxk in xx]

# Bestimme die interpolierten Werte für jedes x
yy = [a[xxi[k]] + b[xxi[k]] * (xx[k] - x[xxi[k]]) + c[xxi[k]] * (xx[k] - x[xxi[k]]) ** 2 + d[xxi[k]] * (xx[k] - x[xxi[k]]) ** 3 for k in range(xx.shape[0])]

plt.figure(1)
plt.grid()
plt.plot(xx, yy, zorder=0, label='spline interpolation')
plt.scatter(x, y, marker='x', color='r', zorder=1, label='measured')
plt.scatter(x_int, y_int, marker='X', color='fuchsia', label='interpolated')
plt.legend()
plt.show()
