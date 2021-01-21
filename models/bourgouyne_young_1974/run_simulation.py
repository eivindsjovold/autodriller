import matplotlib.pyplot as plt

from simulation import simulation


## Drilling parameters
D = 0
D_final = 1000
rho = 1000
rpm = 500
q = 300
wob_init = 30
wob = 100
gp = 10
db = 1
db_init = db
delta_t = 1
h = 0.001
v = 10

time_dict, depth_dict, rop_dict = simulation(D,gp,rho,wob,wob_init,db,db_init,rpm,h,q,v, D_final, delta_t)

plt.figure()
plt.subplot(211)
plt.plot(time_dict, rop_dict, 'r')
plt.xlabel('Time')
plt.ylabel('Rate of Penetration')

plt.subplot(212)
plt.plot(time_dict, depth_dict)
plt.xlabel('Time')
plt.ylabel('Depth')
#plt.show()

