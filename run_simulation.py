import matplotlib.pyplot as plt
from models.simulation import simulation
from models.bourgouyne_young_1974.read_from_case import read_from_case

case = 'WOB'
model = 'test'


## Drilling parameters
#BY
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
#Eckel
v = 0.1  #feet/s
a = 1
b = 1
c = 1
K = 1
k = 1
my = 0.4
a11 = 0.05
a22 = 0.05
a33 = 0.05



formation_change = read_from_case()
time_dict, depth_dict, rop_dict, model_parameters = simulation(depth,gp,rho,wob,wob_init,db,db_init,rpm,h,q,v, depth_final, delta_t, case, formation_change,a,b,c,k,K,my,model,a11,a22,a33)
print(rop_dict)

plt.figure()
plt.subplot(211)
plt.plot(time_dict, rop_dict, 'r')
plt.xlabel('Time')
plt.ylabel('Rate of Penetration')
if case == 'RPM' or 'Q' or 'WOB':
    pass
else:
    plt.subplot(212)
    plt.plot(time_dict, depth_dict)
    plt.xlabel('Time')
    plt.ylabel('Depth')

plt.savefig('models\\figures\\rop_test.png')
plt.show()
