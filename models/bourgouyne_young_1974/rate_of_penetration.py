from numpy import exp

def jet_force(rho,q,v):
    return 0.000515*rho*q*v
    
def f1(a1):
    return exp(2.303*a1)

def f2(D,a2):
    return exp(2.303*a2*(10000-D))

def f3(gp, a3, D):
    return exp(2.303*a3*(D**0.69)*(gp-9))

def f4(a4,D,gp,rho):
    return exp(2.303*a4*D*(gp-rho))

def f5(a5, wob, wob_init, db, db_init):
    wob_diameter = wob/db
    wob_threshold = wob_init/db_init
    return ((wob_diameter-wob_threshold)/(4-wob_threshold))**a5

def f6(a6, rpm):
    return (rpm/60)**a6

def f7(a7, h):
    return exp(-a7*h)

def f8(a8, rho, q, v):
    return (jet_force(rho,q,v)/1000)**a8


def rate_of_penetration(a1,a2,a3,a4,a5,a6,a7,a8,D,gp,rho,wob,wob_init,db,db_init,rpm,h,q,v):
    return f1(a1)*f2(D,a2)*f3(gp, a3, D)*f4(a4,D,gp,rho)*f5(a5, wob, wob_init, db, db_init)*f6(a6, rpm)*f7(a7, h)*f8(a8, rho,q,v)
