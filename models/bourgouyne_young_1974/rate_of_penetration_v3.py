import numpy as np


def jet_force(rho,q,v):
    return 0.000515*rho*q*v
## v = nozzle velocity
    
def f1(a1): #formation strenght
    return np.exp(2.303*a1)

def f2(depth,a2):  
    return np.exp(2.303*a2*(10000-depth))

def f3(pore_pressure_gradient, a3, depth):
    return np.exp(2.303*a3*np.power(depth,0.69)*(pore_pressure_gradient-(9))) #*(0.45359/0.003785))))

def f4(a4,depth,pore_pressure_gradient,rho):
    return np.exp(2.303*a4*depth*(pore_pressure_gradient-rho))

## Rho = equivalent cirulating density

def f5(a5, wob, wob_init, db, db_init,a11):
    wob_diameter = wob/db
    wob_threshold = wob_init/db_init
    temp = ((wob_diameter - wob_threshold)/( 4.0 - wob_threshold))
    return max(0,(np.power(temp , a5) - a11*np.power(temp,((a5+1.33)))))


def f6(a6, rpm, a22):
    return max(0,(np.power(rpm/150, a6) - a22*(np.power(rpm/150,(a6+8)))))


def f7(a7, h):
    #print('rpm:',np.exp(-a7*h))
    return np.exp(-a7*h)


def f8(a8, rho, q, v, a33):
    #print('q:',max(0, (np.power(jet_force(rho,q,v), a8)) - a33*(np.power(jet_force(rho,q,v),a8+1.33))))
    return max(0, 1.5*(np.power(jet_force(rho,q,v), a8)) - a33*(np.power(jet_force(rho,q,v),a8+1.33)))


def isFounder(operational_parameter, parameter_constant, hardness):
    if operational_parameter*parameter_constant*hardness > 150:
        return True
    else:
        return False


def rate_of_penetration_modv3(a1,a2,a3,a4,a5,a6,a7,a8,depth,gp,rho,wob,wob_init,db,db_init,rpm,h,q,v, a11, a22, a33):
    temp = f1(a1)*f5(a5, wob, wob_init, db, db_init,a11)*f6(a6, rpm, a22)*f8(a8, rho,q,v, a33)
    #if temp == 0:
    #    return (abs(q) + abs(wob) + abs(rpm)) - 1.05*(abs(q) + abs(wob) + abs(rpm))
    #else:
    return temp

'''
def rate_of_penetration_modv3(a1,a2,a3,a4,a5,a6,a7,a8,depth,gp,rho,wob,wob_init,db,db_init,rpm,h,q,v, a11, a22, a33, f5_last, f6_last):
    founder_wob = isFounder(wob,a5,a1)
    founder_rpm = isFounder(rpm,a6,a1)

    wob_function = f5(a5, wob, wob_init, db, db_init)
    rpm_function = f6(a6, rpm)

    if founder_wob:
        wob_function = f5_last - 0.06* wob_function
    if founder_rpm:
        rpm_function = f6_last - 0.001* rpm_function

    rop = f1(a1)*f2(depth,a2)*f3(gp, a3, depth)*f4(a4,depth,gp,rho)*wob_function*rpm_function*f7(a7, h)*f8(a8, rho,q,v)
    return rop, wob_function, rpm_function
'''
