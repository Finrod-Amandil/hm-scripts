# -*- coding: utf-8 -*-
"""
Serie 1, Aufgabe 1

@author: IT19ta_WIN10
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-10, 10, 0.001)

#f = x⁵ - 5x⁴ - 30x³ + 110x² + 29x - 105
f = (x ** 5) - (5 * (x ** 4)) - (30 * (x ** 3)) + (110 * (x ** 2)) + (29 * x) - 105

#f' = x⁴ - 20x³ - 90x² + 220x + 29
df = (5 *(x ** 4)) - (20 * (x ** 3)) - (90 * (x ** 2)) + (220 * x) + 29

#F(x) = 1/6x⁶ - x⁵ - 15/2x⁴ + 110/3x³ + 29/2x² - 105x
F = ((1.0 / 6) * (x ** 6)) - (x ** 5) - ((15.0 / 2) * (x ** 4)) + ((110.0 / 3) * (x ** 3)) + ((29.0 / 2) * (x ** 2)) - 105 * x

plt.plot(x, f)
plt.plot(x, df)
plt.plot(x, F)
plt.xlim(-6, 8)
plt.ylim(-1400, 1400)
plt.grid()
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("f(x) = x⁵ - 5x⁴ - 30x³ + 110x² + 29x - 105 with derivative and integral")
plt.legend(["f(x)", "f'(x)", "F(x)"])
plt.show()

