import numpy as np
import matplotlib.pyplot as plt


def rate_of_penetration_eckel(a,b,c,K, k, wob, rpm, q, rho, d_n, my):
    #hardcoded at some specific values for now
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
    return K*(wob**a)*(rpm**b)*((k*q*rho)/(d_n*my))**c
