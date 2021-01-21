import numpy as np


def jet_force(rho,q,v):
    return 0.000515*rho*q*v
    
def f1(a1):
    return np.exp(2.303*a1)

def f2(depth,a2):
    return np.exp(2.303*a2*(10000-depth))

def f3(gp, a3, depth):
    return np.exp(2.303*a3*(depth**0.69)*(gp-9))

def f4(a4,depth,gp,rho):
    return np.exp(2.303*a4*depth*(gp-rho))

def f5(a5, wob, wob_init, db, db_init):
    wob_diameter = wob/db
    wob_threshold = wob_init/db_init
    temp = ((wob_diameter - wob_threshold)/( 4.0 - wob_threshold))
    return np.power(temp , a5)

def f6(a6, rpm):
    return np.power(rpm/60, a6)

def f7(a7, h):
    return np.exp(-a7*h)

def f8(a8, rho, q, v):
    return np.power(jet_force(rho,q,v)/1000, a8)


def rate_of_penetration(a1,a2,a3,a4,a5,a6,a7,a8,depth,gp,rho,wob,wob_init,db,db_init,rpm,h,q,v):
    return f1(a1)*f2(depth,a2)*f3(gp, a3, depth)*f4(a4,depth,gp,rho)*f5(a5, wob, wob_init, db, db_init)*f6(a6, rpm)*f7(a7, h)*f8(a8, rho,q,v)
