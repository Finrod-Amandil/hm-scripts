import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


# Funktionsdefinition:
def f(x, y):
    return np.sin(x) * np.cos(x * 0.5 * y) + x + 0.5 * y - 0.2 * x * y


# Wertebereich definieren:
xmin = 0
xmax = 10
ymin = -5
ymax = 5

[x, y] = np.meshgrid(np.linspace(xmin, xmax), np.linspace(ymin, ymax))
z = f(x, y)

# Wireframe
fig = plt.figure(0)
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z)
plt.title('Wireframe')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

# Colormap
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm)
fig.colorbar(surf)
plt.title('Colormap')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

# Höhenlinien
fig = plt.figure(2)
cont = plt.contour(x, y, z)
fig.colorbar(cont)
plt.title('Höhenlinien')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
