
"""
@author: IT19ta_WIN / zahlesev@students.zhaw.ch
@version: 1.0, 25.01.2021
"""


import numpy as np

"""==================== INPUT ===================="""
A = np.array([[1, -2, 0], [2, 0, 1], [0, -2, 1]], dtype=np.float64)
"""==============================================="""

def qr_decomposition(A_in, iteration_count):
    A = np.copy(A_in)
    P = np.eye(A.shape[0])

    for i in range(iteration_count):
        Q, R = np.linalg.qr(A)
        A = R @ Q
        P = P @ Q

    return A, P


print(np.linalg.eig(A))
print("")
A, P = qr_decomposition(A, 1000)
print(A)

