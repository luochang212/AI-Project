import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math


def calculator(p, q):
    """Cross entropy"""
    return - p * math.log(q)


x = np.linspace(0.01, 1, 100)
z = np.array([[calculator(i, j) for i in x] for j in x])
x, y = np.meshgrid(x, x)


fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.set_xlabel('p')
ax.set_ylabel('q')
ax.set_zlabel('H(p, q)')
ax.plot_surface(x, y, z, cmap=cm.gray)
plt.show()
