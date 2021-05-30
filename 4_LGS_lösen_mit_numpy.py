"""
@author: IT19ta_WIN / zahlesev@students.zhaw.ch
@version: 1.1, 25.01.2021
"""

import numpy as np

"""==================== INPUT ===================="""
A = np.array([[   4,  -1,  -5],
               [-12,   4,  17],
               [ 32, -10, -41]],
              dtype=np.float64)
b = np.array([-5, 19, -39], dtype=np.float64)
"""==============================================="""

print("x = " + str(np.linalg.solve(A, b)))
