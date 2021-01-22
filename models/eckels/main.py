import numpy as np
import matplotlib.pyplot as plt

rpm = 40
wob = 11000
q = 334
a = 0.1
b = 0.11
c = 0.5
K = 0.1
k = 0.1
rho = 1000
d_n = 0.25
my = 0.4
D_final = 10000
delta_t = 1

def rate_of_penetration(a,b,c,K, k, wob, rpm, q, rho, d_n, my):
    return K*(wob**a)*(rpm**b)*((k*q*rho)/(d_n*my))**c

def simulation(a,b,c,K, k, wob, rpm, q, rho, d_n, my, D_final, delta_t):
    D = 0
    t = 0
    depth = []
    t_dict = []
    rate = []
    while D < D_final:
        rop = rate_of_penetration(a,b,c,K, k, wob, rpm, q, rho, d_n, my)
        D += rop*delta_t
        t += delta_t
        rate.append(rop)
        depth.append(D)
        t_dict.append(t)
    return t_dict, depth, rate

def plot(a,b,c,K, k, wob, rpm, q, rho, d_n, my, D_final, delta_t):
    t, depth, rop = simulation(a,b,c,K, k, wob, rpm, q, rho, d_n, my, D_final, delta_t)

    plt.figure()
    plt.subplot(211)
    plt.plot(t,depth)

    plt.subplot(212)
    plt.plot(t,rop)
    plt.show()

    
plot(a,b,c,K, k, wob, rpm, q, rho, d_n, my, D_final, delta_t)


