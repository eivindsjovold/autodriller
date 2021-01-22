import matplotlib.pyplot as plt
from simulation import simulation


## Drilling parameters
depth = 0.0
depth_final = 1000.0
rho = 10.0
rpm = 3.0
q = 30.0
wob_init = 0.6
wob = 5
gp = 1.0
db = 1.0
db_init = db
delta_t = 1.0
h = 0.0000001
v = 10.0

time_dict, depth_dict, rop_dict = simulation(depth,gp,rho,wob,wob_init,db,db_init,rpm,h,q,v, depth_final, delta_t)

plt.figure()
plt.subplot(211)
plt.plot(time_dict, rop_dict, 'r')
plt.xlabel('Time')
plt.ylabel('Rate of Penetration')

plt.subplot(212)
plt.plot(time_dict, depth_dict)
plt.xlabel('Time')
plt.ylabel('Depth')
plt.show()

