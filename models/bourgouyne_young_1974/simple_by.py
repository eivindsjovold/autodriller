import numpy as np

def simple_by(a1,a2,a5,a6,a8,d,rpm,q,wob):
    return a1*np.power(D,a2)*np.power(wob,a5)*np.power(rpm,a6))*np.power(q,a8)
