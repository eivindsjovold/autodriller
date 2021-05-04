import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt



z_np = np.linspace(-10,10,400)
z = torch.from_numpy(z_np)

h = nn.ReLU()
x = h(z)
plt.figure('ReLU')
plt.plot(z_np,x, color = 'orange', label = 'ReLU')
plt.grid(axis = 'x', color ='0.95')
plt.grid(axis = 'y', color ='0.95')
plt.ylim(-10,10)
plt.xlim(-10,10)
plt.legend()
plt.savefig('thesis_fig\\activation_functions\\relu.eps', format = 'eps')
plt.show()