# -*- coding: utf-8 -*-
"""
Serie 1, Aufgabe 2

@author: IT19ta_WIN10 / zahlesev@students.zhaw.ch
"""

import numpy as np
import matplotlib.pyplot as plt

"""
a: array of polynome coefficients
xmin, xmax: x-range to calculate derivative on.

returns:
x: x-values for which function values are provided
p: function values of polynomial
dp: function values of derivative of polynomial
dpint: function values of integral of polynomial
"""
def IT19ta_WIN10_Aufg2(a, xmin, xmax):
    
    if np.shape(a)[0] == 0 or len(np.shape(a)) > 1:
        raise Exception("Array of factors has an invalid shape.")
    
    x = np.arange(xmin, xmax, (abs(xmax - xmin)) / 10000.0)
    
    p = np.full((x.size), 0)
    dp = np.full((x.size), 0)
    pint = np.full((x.size), 0)
    
    for exponent, factor in enumerate(a):
        for index, x_cur in enumerate(x):
            p[index] += factor * (x_cur ** exponent)
        
            dp[index] += factor * exponent * (x_cur ** (exponent - 1))
            
            pint[index] += factor * (1.0 / (exponent + 1)) * (x_cur ** (exponent + 1))

    return(x, p, dp, pint)


""" example for x⁵ - 5x⁴ - 30x³ + 110x² + 29x - 105 over interval [-10, 10]"""
[x, p, dp, pint] = IT19ta_WIN10_Aufg2([-105, 29, 110, -30, -5, 1], -10, 10)

plt.plot(x, p)
plt.plot(x, dp)
plt.plot(x, pint)
plt.xlim(-6, 8)
plt.ylim(-1400, 1400)
plt.grid()
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("f(x) derivative and integral")
plt.legend(["f(x)", "f'(x)", "F(x)"])
plt.show()