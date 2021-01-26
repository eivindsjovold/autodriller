import matplotlib.pyplot as plt
from simulation import simulation
from read_from_case import read_from_case

case = 'changing_formation'


## Drilling parameters
depth = 0.0 #feet
depth_final = 500.0 #feet
rho = 22 
rpm = 600.0
q = 300     #gal/min
wob_init = 0.1 #10^3 lbf/in
wob = 10      #10^3 lbf/in 
gp = 1.0        #lbm/gal
db = 1.0
db_init = db    #bit outer diameter in inches
delta_t = 1/3600 #hrs
h = 0.0001
v = 0.1  #feet/s

formation_change = read_from_case()
time_dict, depth_dict, rop_dict, model_parameters = simulation(depth,gp,rho,wob,wob_init,db,db_init,rpm,h,q,v, depth_final, delta_t, case, formation_change)

plt.figure()
plt.subplot(211)
plt.plot(time_dict, rop_dict, 'r')
plt.xlabel('Time')
plt.ylabel('Rate of Penetration')

plt.subplot(212)
plt.plot(time_dict, depth_dict)
plt.xlabel('Time')
plt.ylabel('Depth')

plt.savefig('figures/rop_test.png')
plt.show()
