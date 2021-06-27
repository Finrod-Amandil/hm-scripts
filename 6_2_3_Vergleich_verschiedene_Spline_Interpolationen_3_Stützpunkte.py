import numpy as np
import matplotlib.pyplot as plt

x = np.array([4, 6, 8], dtype=np.float64)  # Stützpunkte (Knoten) xi
y = np.array([9, 3, 9], dtype=np.float64)  # Stützpunkte (Knoten) yi

x_int = 7.5
i_int = np.max(np.where(x <= x_int))  # Finde die Stützstelle, deren x-Wert am grössten, aber gerade noch kleiner ist als x_int

n = x.shape[0] - 1  # Anzahl Spline-Polynome

xx = np.arange(x[0], x[-1], (x[-1] - x[0]) / 10000)  # Plot-X-Werte
xxi = [np.max(np.where(x <= xxk)) for xxk in xx]  # Bestimme für jeden x-Wert, welches Spline-Polynom gilt

# Allgemeine Spline 3. Grades und deren Ableitungen:
# Si(x) = ai + bi(x - xi) + ci(x - xi)^2 + di(x - xi)^3
# Si'(x) = bi + ci * 2(x - xi) + di * 3(x - xi)^2
# Si''(x) = ci * 2 + di * 6(x - xi)
# Si'''(x) = di * 6

# Natürliche kubische Spline-Interpolation mit 3 Stützstellen
# -----------------------------------------------------------
A_nat = np.array([
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

b_nat = np.array([
    y[0],  # S0(x0) = y0
    y[1],  # S1(x1) = y1
    y[1],  # S0(x1) = y1
    y[2],  # S1(x2) = y2
    0,  # S0'(x1) - S1'(x1) = 0
    0,  # S0''(x1) - S1''(x1) = 0
    0,  # S0''(x0) = 0
    0,  # S1''(x2) = 0
], dtype=np.float64)


abcd_nat = np.linalg.solve(A_nat, b_nat)
a_nat = abcd_nat[0:2]
b_nat = abcd_nat[2:4]
c_nat = abcd_nat[4:6]
d_nat = abcd_nat[6:]

y_int_nat = a_nat[i_int] + b_nat[i_int] * (x_int - x[i_int]) + c_nat[i_int] * (x_int - x[i_int]) ** 2 + d_nat[i_int] * (x_int - x[i_int]) ** 3
yy_nat = [a_nat[xxi[k]] + b_nat[xxi[k]] * (xx[k] - x[xxi[k]]) + c_nat[xxi[k]] * (xx[k] - x[xxi[k]]) ** 2 + d_nat[xxi[k]] * (xx[k] - x[xxi[k]]) ** 3 for k in range(xx.shape[0])]

# ======================================================================================================================

# Periodische kubische Spline-Interpolation mit 3 Stützstellen
# ------------------------------------------------------------
A_prd = np.array([
    # a0 a1 b0 b1 c0 c1 d0 d1
    [1, 0, 0, 0, 0, 0, 0, 0],  # S0(x0) = y0 <=> a0 + b0(x0 - x0) + c0(x0 - x0)^2 + d0(x0 - x0)^3 = y0 <=> a0 = y0 (Spline 0 muss durch (x0, y0) gehen)
    [0, 1, 0, 0, 0, 0, 0, 0],  # S1(x1) = y1 <=> a1 + b1(x1 - x1) + c1(x1 - x1)^2 + d1(x1 - x1)^3 = y1 <=> a1 = y1 (Spline 1 muss durch (x1, y1) gehen)
    [1, 0, x[1] - x[0], 0, (x[1] - x[0]) ** 2, 0, (x[1] - x[0]) ** 3, 0],  # S0(x1) = S1(x1) <=> S0(x1) = y1 <=> a0 + b0(x1 - x0) + c0(x1 - x0)^2 + d0(x1 - x0)^3 = y1 (Spline 0 und 1 müssen sich im Punkt (x1, y1) treffen <=> Spline 0 muss durch (x1, y1) gehen)
    [0, 1, 0, x[2] - x[1], 0, (x[2] - x[1]) ** 2, 0, (x[2] - x[1]) ** 3],  # S1(x2) = y2 <=> a1 + b1(x2 - x1) + c1(x2 - x1)^2 + d1(x2 - x1)^3 = y2 (Spline 1 muss durch (x2, y2) gehen (letzter Stützpunkt))
    [0, 0, 1, -1, 2 * (x[1] - x[0]), 0, 3 * (x[1] - x[0]) ** 2, 0],  # S0'(x1) = S1'(x1) <=> S0'(x1) - S1'(x1) = 0 <=> b0 - b1 + c0 * 2(x1 - x0) - c1 * 2(x1 - x1) + d0 * 3(x1 - x0)^2 - d1 * 3(x1 - x1)^2 = 0 <=> b0 - b1 + c0 * 2(x1 - x0) + d0 * 3(x1 - x0)^2 = 0 (Keine Knicke zwischen S0 und S1)
    [0, 0, 0, 0, 2, -2, 6 * (x[1] - x[0]), 0],  # S0''(x1) = S1''(x1) <=> S0''(x1) - S1''(x1) = 0 <=> c0 * 2 - c1 * 2 + d0 * 6(x1 - x0) - d0 * 6(x1 - x1) = 0 <=> c0 * 2 - c1 * 2 + d0 * 6(x1 - x0) = 0 (Gleiche Krümmung zwischen S0 und S1)
    [0, 0, 1, -1, 0, -2*(x[2]-x[1]), 0, -3*(x[2]-x[1])**2],  # PERIODISCHE SPLINE: S0'(x0) = S1'(x2) <=> S0'(x0) - S1'(x2) = 0 <=> b0 - b1 + c0 * 2(x0 - x0) - c1 * 2(x2 - x1) + d0 * 3(x0 - x0)^2 - d1 * 3(x2 - x1)^2 = 0 <=> b0 - b1 - c1 * 2(x2 - x1) - d1 * 3(x2 - x1)^2 = 0 (Gleiche Steigung am Anfang und Ende)
    [0, 0, 0, 0, 2, -2, 0, -6*(x[2]-x[1])]  # PERIODISCHE SPLINE: S0''(x0) = S1''(x2) <=> S0''(x0) - S1''(x2) = 0 <=> c0 * 2 - c1 * 2 + d0 * 6(x0 - x0) - d1 * 6(x2 -x1) = 0 <=> c0 * 2 - c1 * 2 - d1 * 6(x2 -x1) = 0 (Gleiche Krümmung am Anfang und Ende)
], dtype=np.float64)

b_prd = np.array([
    y[0],  # S0(x0) = y0
    y[1],  # S1(x1) = y1
    y[1],  # S0(x1) = y1
    y[2],  # S1(x2) = y2
    0,  # S0'(x1) - S1'(x1) = 0
    0,  # S0''(x1) - S1''(x1) = 0
    0,  # S0'(x0) - S1'(x2) = 0
    0,  # S0''(x0) - S1''(x2) = 0
], dtype=np.float64)

abcd_prd = np.linalg.solve(A_prd, b_prd)
a_prd = abcd_prd[0:2]
b_prd = abcd_prd[2:4]
c_prd = abcd_prd[4:6]
d_prd = abcd_prd[6:]

y_int_prd = a_prd[i_int] + b_prd[i_int] * (x_int - x[i_int]) + c_prd[i_int] * (x_int - x[i_int]) ** 2 + d_prd[i_int] * (x_int - x[i_int]) ** 3
yy_prd = [a_prd[xxi[k]] + b_prd[xxi[k]] * (xx[k] - x[xxi[k]]) + c_prd[xxi[k]] * (xx[k] - x[xxi[k]]) ** 2 + d_prd[xxi[k]] * (xx[k] - x[xxi[k]]) ** 3 for k in range(xx.shape[0])]

# ======================================================================================================================

# Not-a-knot kubische Spline-Interpolation mit 3 Stützstellen
# -----------------------------------------------------------

# NOT-A-KNOT IST NUR MIT 4+ STÜTZPUNKTEN MÖGLICH!


plt.figure(1)
plt.grid()
plt.plot(xx, yy_nat, zorder=0, label='natural spline')
plt.plot(xx, yy_prd, zorder=0, label='periodic spline')
plt.scatter(x, y, marker='x', color='r', zorder=1, label='measured')
plt.scatter(x_int, y_int_nat, marker='X', color='fuchsia', label='interpolated')
plt.scatter(x_int, y_int_prd, marker='X', color='fuchsia')
plt.legend()
plt.show()
