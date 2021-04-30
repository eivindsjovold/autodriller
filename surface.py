from models.simple_model import rop_multi
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

xax = np.arange(0,300,1)
yax = np.arange(0,300,1)
R = np.sqrt(xax**2 + yax**2)
Z = np.sin(R)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(xax,yax,Z,cmap =cm.coolwarm, linewidth = 0, antaliased = False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()