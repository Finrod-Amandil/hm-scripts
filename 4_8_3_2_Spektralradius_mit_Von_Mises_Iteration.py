
"""
@author: IT19ta_WIN / zahlesev@students.zhaw.ch
@version: 1.0, 25.01.2021
"""


import numpy as np
import matplotlib.pyplot as plt

"""==================== INPUT ===================="""
A = np.array([[13, 6], [-20, -9]], dtype=np.float64)

v0 = np.array([-1, 0]).T
"""==============================================="""

def von_mises_iteration(A_in, v_in, iterations):
    A = np.copy(A_in)
    v = np.copy(v_in)
    eigv = 0

    values = []

    for i in range(iterations):
        v_next = (A @ v) / (np.linalg.norm(A @ v))
        eigv = (v.T @ A @ v) / (v.T @ v)
        v = v_next
        values.append(eigv)

        print("n = " + str(i) + ": λ = " + str(eigv))

    return eigv, v, values


ew, ev, values = von_mises_iteration(A, v0, 40)
print("Grösster Eigenwert / Spektralradius = " + str(ew))
print("Zugehöriger Eigenvektor = " + str(ev))

x = np.arange(0, 40)
deltas = [abs(value - 3) for value in values]

plt.figure(1)
plt.semilogy()
plt.plot(x, deltas)
plt.grid()
plt.show()

