import matplotlib.pyplot as plt
print(plt.__version)
from models.simple_model import rop
rop_dict = []
for i in range(0,200):
    r = rop(i,100)
    rop_dict.append(r)
plt.figure('2')
plt.plot(rop_dict)
plt.show()