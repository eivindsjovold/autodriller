from models.bourgouyne_young_1974.rate_of_penetration import rate_of_penetration
from models.bourgouyne_young_1974.rate_of_penetration_modified import rate_of_penetration_mod
from models.bourgouyne_young_1974.generate_parameter import generate_range
from models.bourgouyne_young_1974.rate_of_penetration_v3 import rate_of_penetration_modv3, f6, f5, f8
from models.eckels.eckel import rate_of_penetration_eckel
from models.eckels.eckel import rate_of_penetration_eckel_individual_founder, rate_of_penetration_eckel_vary_founder, rop_eck
from models.hareland.hareland_model import rop_hareland, area
import numpy as np
import gym
import rop_envs
import matplotlib.pyplot as plt
import random
import math

from stable_baselines3 import DQN


def simulation(depth,gp,rho,wob,wob_init,db,db_init,rpm,h,q,v, depth_final, delta_t, case, formation_change,a,b,c,k,K,my,model,a11,a22,a33):
    ## initialize return dictionaries
    time_dict = []
    depth_dict = []
    rop_dict = []
    rpm_dict = []
    wob_dict = []
    flow_dict = []
    model_parameters = []
    t = 0.0
    
    if model == 'BY':
        print(model)
        if case == 'test_model':
            print(case)
            a1 = 1.5
            a2 = 0
            a3 = a2
            a4 = a2
            a7 = a2
            a5 = 1
            a6 = 0.75
            a8 = 0.1
            rpm = 60
            wob_m = 0
            rpm_m = 0
            rop_m = -999999
            q = 400
            time = 0
            rop = 0
            wob_part = rop
            rpm_part = rop
            wob =111
            rpm = 214
            q = 87
            a11 = 0.005
            a22 = 0.005
            a33 = 0.005
            vec = [1.0, 1.5]
            rop_dict2 = []
            rop_max = 0
            index = 0
            for i in range(0,400):
                rop = rate_of_penetration_modv3(a1,a2,a3,a4,a5,a6,a7,a8,0,gp,rho,wob,wob_init,db,db_init,rpm,h,q,v, a11,a22,a33)
                #rop = f5(a5, wob, wob_init, db, db_init,a11)
                #rop = f6(a6, rpm, a22)
                #rop = f8(a8, rho, q, v, a33)
                rop_dict.append(rop)
                if rop > rop_max:
                    index = q
                    rop_max = rop
                    print('index', index)
                    print('max', rop_max)
                time += delta_t
                depth += rop*delta_t
                depth_dict.append(depth)
                time_dict.append(time)
                depth_dict.append(depth)
            #plt.figure()
            #plt.plot(rop_dict)
            #plt.show()        
        
        
        
        
        
        elif case == 'nested_loop':
            print(case)
            ##initialize parameters
            a1,a2,a3,a4,a5,a6,a7,a8 = generate_range()
            a2 = 0
            a3 = 0
            a4 = 0
            a1 = 1.5
            a5 = 1
            a6 = 0.6
            a7 = 0
            a8 = 0.3
            #a1 = 1.9
            #a5 = 2
            #a6 = 1
            #a8 = 0.6
            a1 = random.uniform(1.0,1.9)
            a5 = 2
            a6 = 1
            a8 = 0.3
            a1 = 1.6
            #a1 = formation_change[1][1]
            #a3 = formation_change[3][1]
            #a4 = formation_change[4][1]
            #a5 = formation_change[5][1]
            #a6 = formation_change[6][1]
            #a7 = formation_change[7][1]
            #a8 = formation_change[8][1]
            #model_parameters = np.array([a1,a2,a3,a4,a5,a6,a7,a8])
            #while depth < depth_final:
            wob_m = 0
            rpm_m = 0
            rop_m = -999999
            for rpm in range(1,300):
                for wob in range(1,100):
                    q = 400
                    rop = rate_of_penetration_modv3(a1,a2,a3,a4,a5,a6,a7,a8,0,gp,rho,wob,wob_init,db,db_init,rpm,h,q,v, a11,a22,a33)
                    if rop > rop_m:
                        rop_m = rop
                        wob_m = wob
                        rpm_m = rpm
                    depth += rop*delta_t
                    t += delta_t
                    depth_dict.append(depth)
                    rop_dict.append(rop)
                    rpm_dict.append(rpm)
                    time_dict.append(t)
            print(rop_m,'wob:',wob_m,'rpm:',rpm_m)
        else:
            print('no case')
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
        print(model)
        if case == 'WOB':
            wob_dict=[]
            print(case)
            rop_ls = []
            depth = 0
            depth_final = 40
            model_parameters = [a,b,c,k,K]
            rpm = 115
            wob = 50
            q = 130
            for i in range(0,400):#while depth < depth_final:
                rop = rate_of_penetration_eckel_vary_founder(a,b,c,0.5, k, wob, rpm, q, rho, db, my, a11, a22, a33)
                depth += rop*delta_t
                t += delta_t
                rop_dict.append(rop)
                depth_dict.append(rop)
                flow_dict.append(q)
                wob_dict.append(wob)
                rpm_dict.append(rpm)
                time_dict.append(rpm)
            max_value = -9999
            rop_max = -9999
            for i in range(len(rop_dict)):
                if rop_dict[i] > rop_max:
                    print(rop_dict[i], flow_dict[i])
                    rop_max = rop_dict[i]
                    max_value = flow_dict[i]
            print('max wob value:',  max_value)
            plt.figure('ROP-RPM relationship')
            plt.title('ROP-RPM relationship')
            plt.plot(rop_dict, 'b')
            plt.xlabel('RPM[rev/min]')
            plt.ylabel('Rate of Penetration[ft/hr]')
            #plt.savefig('thesis_fig\\rpm_rop_curve_eckel.eps', format = 'eps')

        elif case == 'RPM':
            print(case)
            rop_ls = []
            depth = 0
            depth_final = 100
            model_parameters = [a,b,c,k,K]
            for rpm in range(0,500):#while depth < depth_final:
                rop = rate_of_penetration_eckel(a,b,c,0.234, k, 30, rpm, 300, rho, db, my, a11, a22, a33)
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
                rop = rate_of_penetration_eckel(a,b,c,0.234, k, 30, 35, q, rho, db, my, a11, a22, a33)
                depth += rop*delta_t
                t += delta_t
                rop_ls.append(rop)
                depth_dict.append(depth)
                rop_dict.append(rop)
                time_dict.append(t)
                print('q: ', q, 'rop: ', rop)
        elif case == 'max':
            print(case)
            a11 = 0.005
            a22 = 0.005
            a33 = 0.00005
            a = 0.1
            b = 0.11
            c = 0.5
            k = 0.1
            rho = 1000
            d_n = 0.25
            my = 0.4
            ub_k = 0.3
            lb_k = 0.1
            K = 0.3
            print(case)
            rop_ls = []
            rop_opt = 0
            counter = 0
            for wob in range(1,100):
                for rpm in range(1,100):
                    for q in range(1,300):
                        rop = rate_of_penetration_eckel(a,b,c,K, k, rpm, wob, q, rho, db, my, a11, a22, a33)
                        counter += 1
                        if rop > rop_opt:
                            wob_opt = wob
                            rpm_opt = rpm
                            q_opt = q
                            rop_opt = rop
                        if counter % 1000000 == 0:
                            print('iterations:', counter, '/125 000 000')
            print('wob:',wob_opt,'rpm:',rpm_opt,'q:',q_opt,'rop:',rop_opt)

        else:
            print('no case')
            rop_ls = []
            q = 150
            wob = 35
            rpm = 35
            depth = 0
            depth_final = 100
            model_parameters = [a,b,c,k,K]
            K = 0.02
            for i in range(0,1):#while depth < depth_final:
                rop = rate_of_penetration_eckel(a,b,c,K, k, 30, 35, q, rho, db, my, a11, a22, a33)
                depth += rop*delta_t
                t += delta_t
                rop_ls.append(rop)
                depth_dict.append(depth)
                rop_dict.append(rop)
                time_dict.append(t)
                print('a: ', a, 'rop: ', rop)
            plt.figure()
            plt.plot(rop_dict)
            plt.show()

    elif model == 'test':
        print(model)
        if case == 'RPM':
            print(case)
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
            print(case)
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
            print(case)
            a1 = formation_change[1][1]
            a2 = formation_change[2][1]
            a3 = formation_change[3][1]
            a4 = 0#formation_change[4][1]
            a5 = formation_change[5][1]
            a6 = formation_change[6][1]
            a7 = 0#formation_change[7][1]
            a8 = formation_change[8][1]
            for q in range(0,1000):
                rop = rate_of_penetration_mod(a1,a2,a3,a4,a5,a6,a7,a8,depth,gp,rho,wob,wob_init,db,db_init,rpm,h,q,v, a11, a22, a33)
                t += delta_t
                time_dict.append(t)
                rop_dict.append(rop)

    elif model =='agent':
        print(model)
        if case =='dqn':
            print(case)
            env = gym.make('eckel-disc-v0')
            dqnModel = DQN.load('trained_agents\\dqn_eckel_mono_116.52')
            obs = env.reset()
            action_log = []
            while obs[0] < depth_final:
                action, _states = dqnModel.predict(obs, deterministic=True)
                action_log.append(action)
                obs, reward, done, info = env.step(action)
                depth += obs[0]*delta_t
                obs[0] = depth
                t += delta_t
                time_dict.append(t)
                rop_dict.append(obs[1])
            print(action_log)                  

    elif model == 'hareland':
        print(model)
        a = 1
        b = 0.6
        c = 0.6
        rpm = 90
        sigma = 100000 #40 MPa
        db = 20
        dc = 15
        num_cutters = 10
        alpha = 0
        tht = math.pi/2
        time_dict = []
        depth_dict  = []
        rop_dict  = []
        model_parameters  = []
        for wob in range(1,10):
            rop = rop_hareland(a,b,c,rpm,wob,num_cutters,db,alpha,tht,dc,sigma)
            rop_dict.append(rop)
        plt.figure()
        plt.plot(rop_dict)
        plt.show()
    return time_dict, depth_dict, rop_dict, model_parameters
    
