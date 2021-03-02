from models.bourgouyne_young_1974.rate_of_penetration import rate_of_penetration
from models.bourgouyne_young_1974.rate_of_penetration_modified import rate_of_penetration_mod
from models.bourgouyne_young_1974.generate_parameter import generate_range
from models.eckels.eckel import rate_of_penetration_eckel
import numpy as np


def simulation(depth,gp,rho,wob,wob_init,db,db_init,rpm,h,q,v, depth_final, delta_t, case, formation_change,a,b,c,k,K,my,model,a11,a22,a33):
    ## initialize return dictionaries
    time_dict = []
    depth_dict = []
    rop_dict = []
    rpm_dict = []
    model_parameters = []
    t = 0.0
    
    if model == 'BY':
        if case == 'uniform':
            ##initialize parameters
            #a1,a2,a3,a4,a5,a6,a7,a8 = generate_range()
            a1 = formation_change[1][1]
            a2 = formation_change[2][1]
            a3 = formation_change[3][1]
            a4 = formation_change[4][1]
            a5 = formation_change[5][1]
            a6 = formation_change[6][1]
            a7 = formation_change[7][1]
            a8 = formation_change[8][1]
            model_parameters = np.array([a1,a2,a3,a4,a5,a6,a7,a8])
            #while depth < depth_final:
            for q in range(5,2500):
                rop = rate_of_penetration_mod(a1,a2,a3,a4,a5,a6,a7,a8,depth,gp,rho,wob,wob_init,db,db_init,rpm,h,q,v, a11, a22, a33)
                depth += rop*delta_t
                t += delta_t
                depth_dict.append(depth)
                rop_dict.append(rop)
                rpm_dict.append(rpm)
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
        if case == 'WOB':
            print(case)
            rop_ls = []
            depth = 0
            depth_final = 100
            model_parameters = [a,b,c,k,K]
            for wob in range(0,500):#while depth < depth_final:
                rop = rate_of_penetration_eckel(a,b,c,K, k, wob, 10, 300, rho, db, my, a11, a22, a33)
                print(' rop: ', rop)
                depth += rop*delta_t
                t += delta_t
                rop_ls.append(rop)
                #if rop_ls[wob] < rop_ls[wob-1]:
                #    print(rop_ls[wob])
                #    print(wob)
                #    break
                
                #depth_dict.append(depth)
                #rop_dict.append(rop)
                #time_dict.append(t)
        elif case == 'RPM':
            print(case)
            rop_ls = []
            depth = 0
            depth_final = 100
            model_parameters = [a,b,c,k,K]
            for rpm in range(0,500):#while depth < depth_final:
                rop = rate_of_penetration_eckel(a,b,c,K, k, 30, rpm, 300, rho, db, my, a11, a22, a33)
                depth += rop*delta_t
                t += delta_t
                rop_ls.append(rop)
                depth_dict.append(depth)
                rop_dict.append(rop)
                time_dict.append(t)
                print(' rop: ', rop, 'Rpm:' , rpm)
        elif case == 'Q':            
            print(case)
            rop_ls = []
            depth = 0
            depth_final = 100
            model_parameters = [a,b,c,k,K]
            for q in range(0,500):#while depth < depth_final:
                rop = rate_of_penetration_eckel(a,b,c,K, k, 30, 35, q, rho, db, my, a11, a22, a33)
                depth += rop*delta_t
                t += delta_t
                rop_ls.append(rop)
                depth_dict.append(depth)
                rop_dict.append(rop)
                time_dict.append(t)
                print('q: ', q, 'rop: ', rop)
    elif model == 'test':
        if case == 'RPM':
            a1 = formation_change[1][1]
            a2 = formation_change[2][1]
            a3 = formation_change[3][1]
            a4 = 0#formation_change[4][1]
            a5 = formation_change[5][1]
            a6 = formation_change[6][1]
            a7 = 0#formation_change[7][1]
            a8 = formation_change[8][1]
            for i in range(0,1000):
                rop = rate_of_penetration_mod(a1,a2,a3,a4,a5,a6,a7,a8,depth,gp,rho,wob,wob_init,db,db_init,rpm,h,q,v, a11, a22, a33)
                rpm += 1
                t += delta_t
                time_dict.append(t)
                rop_dict.append(rop)

        if case == 'WOB':
            a1 = formation_change[1][1]
            a2 = formation_change[2][1]
            a3 = formation_change[3][1]
            a4 = 0#formation_change[4][1]
            a5 = formation_change[5][1]
            a6 = formation_change[6][1]
            a7 = 0#formation_change[7][1]
            a8 = formation_change[8][1]
            for i in range(0,1000):
                rop = rate_of_penetration_mod(a1,a2,a3,a4,a5,a6,a7,a8,depth,gp,rho,wob,wob_init,db,db_init,rpm,h,q,v, a11, a22, a33)
                wob += 1.5
                t += delta_t
                time_dict.append(t)
                rop_dict.append(rop)                  
                print(wob)
        if case == 'Q':
            a1 = formation_change[1][1]
            a2 = formation_change[2][1]
            a3 = formation_change[3][1]
            a4 = 0#formation_change[4][1]
            a5 = formation_change[5][1]
            a6 = formation_change[6][1]
            a7 = 0#formation_change[7][1]
            a8 = formation_change[8][1]
            for i in range(0,1000):
                rop = rate_of_penetration_mod(a1,a2,a3,a4,a5,a6,a7,a8,depth,gp,rho,wob,wob_init,db,db_init,rpm,h,q,v, a11, a22, a33)
                q += 10
                t += delta_t
                time_dict.append(t)
                rop_dict.append(rop)
    
    return time_dict, depth_dict, rop_dict, model_parameters
    
