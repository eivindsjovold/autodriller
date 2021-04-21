import numpy as np
import matplotlib.pyplot as plt


def rate_of_penetration_eckel(a,b,c,K, k, wob, rpm, q, rho, d_n, my, a11, a22, a33):
    #hardcoded at some specific values for now
    a = 0.1
    b = 0.11
    c = 0.5
    k = 0.1
    rho = 1000
    d_n = 0.25
    my = 0.4
    return 2*(K*np.power(wob,a)*np.power(rpm,b)*np.power(((k*q*rho)/(d_n*my)),c) - a11*np.power(wob,2) - a22*np.power(rpm,2) - a33*np.power(q,2.5))

def rate_of_penetration_eckel_individual_founder(a,b,c,K, k, wob, rpm, q, rho, d_n, my, a11, a22, a33):
    a = 1
    b = 0.75
    c = 0.5
    k = 0.1
    rho = 1000
    d_n = 0.25
    my = 0.4
    part_1 = np.power(wob,a)
    part_2 = a11*np.power(wob,2)
    return  K*max(0,(np.power(wob,a) - a11*np.power(wob,2)))*max(0,(np.power(rpm,b) - a22*np.power(rpm,1.66)))* 1/100*max(0,(np.power(((k*q*rho)/(d_n*my)),c) - a33 * np.power(((k*q*rho)/(d_n*my)),0.9)))


def rate_of_penetration_eckel_vary_founder(a,b,c,K, k, wob, rpm, q, rho, d_n, my, a11, a22, a33):
    a = 1
    b = 0.75
    c = 0.5
    k = 0.1
    rho = 1000
    d_n = 0.25
    my = 0.4
    part_1 = np.power(wob,a)
    part_2 = a11*np.power(wob,2)
    return K*max(0,(np.power(wob,a) - (a11 + K/100)*np.power(wob,2)))*max(0,(np.power(rpm,b) - a22*np.power(rpm,1.66)))* 1/100*max(0,(np.power(((k*q*rho)/(d_n*my)),c) - a33 * np.power(((k*q*rho)/(d_n*my)),0.9)))