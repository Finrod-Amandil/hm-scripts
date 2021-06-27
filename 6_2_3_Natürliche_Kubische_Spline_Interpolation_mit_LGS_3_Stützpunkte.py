import numpy as np
import matplotlib.pyplot as plt

x = np.array([4, 6, 8], dtype=np.float64)  # Stützpunkte (Knoten) xi
y = np.array([6, 3, 9], dtype=np.float64)  # Stützpunkte (Knoten) yi
x_int = 7.5  # Zu interpolierender Wert

n = x.shape[0] - 1  # Anzahl Spline-Polynome
dim = 3

print('{} Spline-Polynome {}. Grades (je {} Koeffizienten) => {} * {} = {} Unbekannte => Es braucht {} Gleichungen.'.format(n, dim, dim+1, n, dim+1, (dim+1)*n, (dim+1)*n))

# Allgemeine Spline 3. Grades und deren Ableitungen:
# Si(x) = ai + bi(x - xi) + ci(x - xi)^2 + di(x - xi)^3
# Si'(x) = bi + ci * 2(x - xi) + di * 3(x - xi)^2
# Si''(x) = ci * 2 + di * 6(x - xi)
# Si'''(x) = di * 6

# Natürliche kubische Spline-Interpolation mit 3 Stützstellen
# -----------------------------------------------------------
A = np.array([
   # a0 a1 b0 b1 c0 c1 d0 d1
    [1, 0, 0, 0, 0, 0, 0, 0],  # S0(x0) = y0 <=> a0 + b0(x0 - x0) + c0(x0 - x0)^2 + d0(x0 - x0)^3 = y0 <=> a0 = y0 (Spline 0 muss durch (x0, y0) gehen)
    [0, 1, 0, 0, 0, 0, 0, 0],  # S1(x1) = y1 <=> a1 + b1(x1 - x1) + c1(x1 - x1)^2 + d1(x1 - x1)^3 = y1 <=> a1 = y1 (Spline 1 muss durch (x1, y1) gehen)
    [1, 0, x[1]-x[0], 0, (x[1]-x[0])**2, 0, (x[1]-x[0])**3, 0],  # S0(x1) = S1(x1) <=> S0(x1) = y1 <=> a0 + b0(x1 - x0) + c0(x1 - x0)^2 + d0(x1 - x0)^3 = y1 (Spline 0 und 1 müssen sich im Punkt (x1, y1) treffen <=> Spline 0 muss durch (x1, y1) gehen)
    [0, 1, 0, x[2]-x[1], 0, (x[2]-x[1])**2, 0, (x[2]-x[1])**3],  # S1(x2) = y2 <=> a1 + b1(x2 - x1) + c1(x2 - x1)^2 + d1(x2 - x1)^3 = y2 (Spline 1 muss durch (x2, y2) gehen (letzter Stützpunkt))
    [0, 0, 1, -1, 2*(x[1]-x[0]), 0, 3*(x[1]-x[0])**2, 0],  # S0'(x1) = S1'(x1) <=> S0'(x1) - S1'(x1) = 0 <=> b0 - b1 + c0 * 2(x1 - x0) - c1 * 2(x1 - x1) + d0 * 3(x1 - x0)^2 - d1 * 3(x1 - x1)^2 = 0 <=> b0 - b1 + c0 * 2(x1 - x0) + d0 * 3(x1 - x0)^2 = 0 (Keine Knicke zwischen S0 und S1)
    [0, 0, 0, 0, 2, -2, 6*(x[1]-x[0]), 0],  # S0''(x1) = S1''(x1) <=> S0''(x1) - S1''(x1) = 0 <=> c0 * 2 - c1 * 2 + d0 * 6(x1 - x0) - d0 * 6(x1 - x1) = 0 <=> c0 * 2 - c1 * 2 + d0 * 6(x1 - x0) = 0 (Gleiche Krümmung zwischen S0 und S1)
    [0, 0, 0, 0, 2, 0, 0, 0],              # NATÜRLICHE SPLINE: S0''(x0) = 0 <=> c0 * 2 + d0 * 6(x0 - x0) = 0 <=> c0 * 2 = 0 (Krümmung in Knoten x0 soll 0 sein)
    [0, 0, 0, 0, 0, 2, 0, 6*(x[2]-x[1])]   # NATÜRLICHE SPLINE: S1''(x2) = 0 <=> c1 * 2 + d1 * 6(x2 - x1) = 0 (Krümmung in Knoten x2 soll 0 sein)
], dtype=np.float64)

b = np.array([
    y[0],  # S0(x0) = y0
    y[1],  # S1(x1) = y1
    y[1],  # S0(x1) = y1
    y[2],  # S1(x2) = y2
    0,  # S0'(x1) - S1'(x1) = 0
    0,  # S0''(x1) - S1''(x1) = 0
    0,  # S0''(x0) = 0
    0,  # S1''(x2) = 0
], dtype=np.float64)

print('\nLöse LGS Ax = b mit')
print('A = \n{}'.format(A))
print('b = {}'.format(b))

print('LGS wird gelöst...\n')

abcd = np.linalg.solve(A, b)
a = abcd[0:2]
b = abcd[2:4]
c = abcd[4:6]
d = abcd[6:]

print('x = {}'.format(abcd))
print('a = {}'.format(a))
print('b = {}'.format(b))
print('c = {}'.format(c))
print('d = {}'.format(d))

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
