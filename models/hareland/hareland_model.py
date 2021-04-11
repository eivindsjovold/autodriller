import numpy as np
import math


def area(dc,w_mech,tht,sigma):
    return np.power(dc/2, 2) * math.acos((1-((4*w_mech)/(math.pi*math.cos(tht)*np.power(dc,2)*sigma))) - np.sqrt((2*w_mech/(math.pi*math.cos(tht)*sigma)) - (4*np.power(w_mech,2)/(math.pi*math.cos(tht)*sigma)))) * ((dc/2) - w_mech/(math.pi*math.cos(tht)*dc*sigma))

def area_v(diameter_cutter, w_mech,tht,sigma):
    P = (2*w_mech)/(math.pi*diameter_cutter*sigma)
    print((2*w_mech)/(math.pi*diameter_cutter*sigma))
    part_1 = np.power(diameter_cutter*0.5,2)
    part_2 = (1 - (2*P) / (math.cos(tht)*diameter_cutter) )
    part_3 =  (diameter_cutter*P)/(math.cos(tht)) - ( (np.power(P,2))/np.power(math.cos(tht), 2)) 
    part_4 = (diameter_cutter*P)/(2*math.cos(tht))
    print(part_1, part_2, part_3, part_4)
    area = part_1*part_2 - np.sqrt(part_3)*part_4
    return area

def rop_hareland(a,b,c,rpm,wob,num_cutters,diameter_bit,alpha,tht,dc,sigma):
    w_mech = wob/num_cutters
    av = area_v(dc,w_mech,tht,sigma)
    normalization = a/(np.power(rpm,b)*np.power(wob,c))
    rop = normalization*((14.14*num_cutters*rpm)/diameter_bit)*math.cos(alpha)*math.sin(tht)
    print('av:', av, 'rop:', rop)
    return rop