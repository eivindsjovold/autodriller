from models.bourgouyne_young_1974.rate_of_penetration import rate_of_penetration
from models.bourgouyne_young_1974.generate_parameter import generate_range
from models.eckels.eckel import rate_of_penetration_eckel
import numpy as np


def simulation(depth,gp,rho,wob,wob_init,db,db_init,rpm,h,q,v, depth_final, delta_t, case, formation_change,a,b,c,k,K,my,model):
    ## initialize return dictionaries
    time_dict = []
    depth_dict = []
    rop_dict = []
    model_parameters = []
    t = 0.0
    
    if model == 'BY':

        if case == 'uniform':
            ##initialize parameters
            a1,a2,a3,a4,a5,a6,a7,a8 = generate_range()
            model_parameters = np.array([a1,a2,a3,a4,a5,a6,a7,a8])
            while depth < depth_final:
                rop = rate_of_penetration(a1,a2,a3,a4,a5,a6,a7,a8,depth,gp,rho,wob,wob_init,db,db_init,rpm,h,q,v)
                depth += rop*delta_t
                t += delta_t
                depth_dict.append(depth)
                rop_dict.append(rop)
                time_dict.append(t)
        else:
            for i in range(0,len(formation_change[0])):
                depth_final = formation_change[0][i]
                while depth < depth_final:
                    rop = rate_of_penetration(formation_change[1][i],formation_change[2][i],formation_change[3][i],formation_change[4][i],formation_change[5][i],formation_change[6][i],formation_change[7][i],formation_change[8][i],depth,gp,rho,wob,wob_init,db,db_init,rpm,h,q,v)
                    depth += rop*delta_t
                    t += delta_t
                    depth_dict.append(depth)
                    rop_dict.append(rop)
                    time_dict.append(t)
    elif model == 'eckel':
        depth = 0
        depth_final = 1000
        model_parameters = [a,b,c,k,K]
        while depth < depth_final:
            rop = rate_of_penetration_eckel(a,b,c,K, k, wob, rpm, q, rho, db, my)
            depth += rop*delta_t
            t += delta_t
            depth_dict.append(depth)
            rop_dict.append(rop)
            time_dict.append(t)
            

    return time_dict, depth_dict, rop_dict, model_parameters
    
