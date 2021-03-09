import numpy as np
import matplotlib.pyplot as plt


def rate_of_penetration_eckel(a,b,c,K, k, wob, rpm, q, rho, d_n, my, a11, a22, a33):
    #hardcoded at some specific values for now
    a = 0.1
    b = 0.11
    c = 0.5
    K = 0.1
    k = 0.1
    rho = 1000
    d_n = 0.25
    my = 0.4
    return K*np.power(wob,a)*np.power(rpm,b)*np.power(((k*q*rho)/(d_n*my)),c) - a11*np.power(wob,2) - a22*np.power(rpm,2)
