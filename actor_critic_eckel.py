import gym
import rop_envs
import csv
from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
#import tensorflow as tf
import numpy as np
#import tensorboard



env = gym.make('eckel-v0')
vec_env = make_vec_env('eckel-v0', n_envs=30)
env2 = gym.make('eckel-v1')
vec_env2 = make_vec_env('eckel-v1', n_envs = 30)
test_env = gym.make('eckel-test-v0')
test_env2 =  gym.make('eckel-test-v1')


case = 'test_eckelenv2'
savestring = 'trained_agents\\for_thesis_eckel2_26_april'
loadstring = 'trained_agents\\for_thesis_eckel2_26_april'#'agents_thesis\\optimum_23_april_eckelenv2'

wob_dict = []
rpm_dict = []
q_dict = []
iteration = []
reward = []
split = []
time_dict = []
rop_dict = []
depth_dict = []
time = 0
flow = []
drillability = []
delta_t = 1/3600

if case == 'train':
    model = A2C(MlpPolicy, vec_env, verbose = 1)#, tensorboard_log='./tensorboard_logs')
    model.learn(total_timesteps = 1000000)#, tb_log_name='session_eckelV2')
    model.save(savestring)


elif case =='load':
    model = A2C.load(loadstring)
    counter = 0
    wob = 10
    rpm = 10
    q = 10
    rop = 0
    depth = 0
    for i in range(0,10):
        done = False
        obs = env2.reset()
        obs = np.array([wob,rpm,q,rop], dtype = np.float32)
        while not done:
            counter += 1
            time += 1/3600
            action, _states = model.predict(obs)
            obs, rewards, done, info = env2.step(action)
            wob = obs[0]
            rpm = obs[1]
            q = obs[2]
            rop = obs[3]
            depth += rop*delta_t
            rop_dict.append(rop)
            time_dict.append(time)
            wob_dict.append(wob)
            rpm_dict.append(rpm)
            q_dict.append(q)
            depth_dict.append(depth)
            reward.append(rewards)
    fig, ax = plt.subplots(5)
    fig.suptitle('RL agent inspired by extremum seeking')
    ax[0].plot(time_dict, rop_dict)
    ax[0].set_title('ROP[ft/hr]')
    ax[1].plot(time_dict, wob_dict)
    ax[2].plot(time_dict, rpm_dict)
    ax[3].plot(time_dict,q_dict)
    
    #plt.savefig('figures\\BY_load_agent')
    plt.show()

elif case == 'train_more':
    for i in range(10):
        print('training session:',i)    
        model = A2C.load(loadstring, verbose = 1)
        model.set_env(vec_env2)
        model.learn(total_timesteps=1000000)
        model.save(savestring)

elif case == 'test_eckelenv1':
    model = A2C.load(loadstring)
    counter = 0
    wob = 10
    rpm =10
    q = 10
    rop = 0
    drillability_vector = [1.4, 0.4324315229681549, 0.6132226175113474, 0.21337671158654828, 0.3319251595736624, 0.31953446975649136, 0.7508770701029669, 0.23063047208802645, 0.47994829875469297, 0.5043919216105951, 0.5351330724339952, 0.32804166681464336, 0.4832090904711962, 0.8732485278109159, 0.5972075094846072, 0.5966112088465494, 0.7782560909631269, 0.6975594101438636, 0.5999282569783442, 0.502991005323338]#, 0.025, 0.9, 0.025]
    formation_length = [244.80946729588345,198.25424579465297,93.06308453533359,146.3732757490667,81.20430179344106,59.31320788619936,200.5636703517305,57.852385884314984,213.28277380654515,114.01947154140029,100.6407360386026,94.4836721604013,85.98994971442879,177.49974259383194,128.5012524687108,198.54914389015798,78.71203996352594,77.34619080434445,123.33890356019359,59.55626904251954]#,169.8487388916136,147.95389419280323,148.84358204029786]
    depth = 0
    for i in range(len(drillability_vector)):
        done = False
        obs = test_env.reset(drillability_vector[i], formation_length[i])
        obs = np.array([wob,rpm,q,rop], dtype = np.float32)
        while not done:
            counter += 1
            time += 1/3600
            action, _states = model.predict(obs)
            obs, rewards, done, info = test_env.step(action)
            wob = obs[0]
            rpm = obs[1]
            q = obs[2]
            rop = obs[3]
            depth += rop*delta_t
            rop_dict.append(rop)
            time_dict.append(time)
            wob_dict.append(wob)
            rpm_dict.append(rpm)
            q_dict.append(q)
            depth_dict.append(depth)
            reward.append(rewards)
    fig, ax = plt.subplots(5)
    fig.suptitle('RL agent inspired by extremum seeking')
    ax[0].plot(time_dict, rop_dict)
    ax[0].set_title('ROP[ft/hr]')
    ax[1].plot(time_dict, wob_dict)
    ax[1].hlines(100,0,time_dict[-1],colors= 'orange',linestyles='dashed',label='Optimal WOB')
    ax[2].plot(time_dict, rpm_dict)
    ax[2].hlines(141,0,time_dict[-1],colors= 'orange',linestyles='dashed',label='Optimal RPM')
    ax[3].plot(time_dict,q_dict)
    ax[3].hlines(130,0,time_dict[-1],colors= 'orange',linestyles='dashed',label='Optimal q')
    ax[4].plot(time_dict,depth_dict)
    plt.show()

    plt.figure('driller plot')
    plt.plot(rop_dict, depth_dict)
    plt.show()
elif case == 'test_eckelenv2':
    loadstring = 'agents_thesis\\optimum_23_april_eckelenv2'
    model = A2C.load(loadstring)
    counter = 0
    wob = 10
    rpm =10
    q = 10
    rop = 0
    drillability_vector = [0.26767342123727805, 0.4324315229681549, 0.6132226175113474, 0.21337671158654828, 0.3319251595736624, 0.31953446975649136, 0.7508770701029669, 0.23063047208802645, 0.47994829875469297, 0.5043919216105951, 0.5351330724339952, 0.32804166681464336, 0.4832090904711962, 0.8732485278109159, 0.5972075094846072, 0.5966112088465494, 0.7782560909631269, 0.6975594101438636, 0.5999282569783442, 0.502991005323338, 0.025, 0.9, 0.025]
    #drillability_vector = [0.4, 0.5, 0.6, 0.7]
    formation_length = [244.80946729588345,198.25424579465297,93.06308453533359,146.3732757490667,81.20430179344106,59.31320788619936,200.5636703517305,57.852385884314984,213.28277380654515,114.01947154140029,100.6407360386026,94.4836721604013,85.98994971442879,177.49974259383194,128.5012524687108,198.54914389015798,78.71203996352594,77.34619080434445,123.33890356019359,59.55626904251954,169.8487388916136,147.95389419280323,148.84358204029786]
    split = []
    depth = 0
    print(len(split),len(drillability_vector))
    for i in range(len(drillability_vector)):
        done = False
        obs = test_env2.reset(drillability_vector[i], formation_length[i])
        #obs = np.array([wob,rpm,q,rop], dtype = np.float32)
        while not done:
            counter += 1
            time += 1/3600
            action, _states = model.predict(obs)
            obs, rewards, done, info = test_env2.step(action)
            wob = obs[0]
            rpm = obs[1]
            q = obs[2]
            rop = obs[3]
            depth += rop*delta_t
            rop_dict.append(rop)
            time_dict.append(time)
            wob_dict.append(wob)
            rpm_dict.append(rpm)
            q_dict.append(q)
            depth_dict.append(depth)
            reward.append(rewards)
            drillability.append(drillability_vector[i])
        split.append(time)

    opt =[]
    wob_opt = []
    rpm_opt = []
    flow_opt = []
    rop_opt = []
    drillability_vector = []
    with open('optimal_values.csv', mode = 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter = ',')
        for row in csv_reader:
            drillability_vector
            drillability_vector.append(float(row[1]))
            wob_opt.append(int(row[1]))
            rpm_opt.append(int(row[2]))
            flow_opt.append(int(row[3]))
            rop_opt.append(float(row[4]))
          
    fig, ax = plt.subplots(6)
    fig.suptitle('RL agent interacting with drilling simulation')
    ax[0].plot(time_dict, rop_dict)
    ax[0].hlines(rop_opt[0],0,split[0],colors= 'orange',linestyles='dashed',label='Optimal WOB')
    for i in range(1,len(split)):
        ax[0].hlines(rop_opt[i],split[i-1],split[i],colors= 'orange',linestyles='dashed',label='Optimal WOB')
    ax[0].set_ylabel('ROP[ft/hr]')
    ax[1].plot(time_dict, wob_dict)
    ax[1].set_ylabel('WOB[klbs]')
    ax[1].hlines(wob_opt[0],0,split[0],colors= 'orange',linestyles='dashed',label='Optimal WOB')
    for i in range(1,len(split)):
        ax[1].hlines(wob_opt[i],split[i-1],split[i],colors= 'orange',linestyles='dashed',label='Optimal WOB')
    ax[2].plot(time_dict, rpm_dict)
    ax[2].set_ylabel('RPM[rev/min]')
    ax[2].hlines(rpm_opt[0],0,split[0],colors= 'orange',linestyles='dashed',label='Optimal RPM')
    for i in range(1,len(split)):
        ax[2].hlines(rpm_opt[i],split[i-1],split[i],colors= 'orange',linestyles='dashed',label='Optimal WOB')
    ax[3].plot(time_dict,q_dict)
    ax[3].set_ylabel('Q[gal/min]')
    ax[3].hlines(flow_opt[0],0,split[0],colors= 'orange',linestyles='dashed',label='Optimal q')
    for i in range(1,len(split)):
        ax[3].hlines(flow_opt[i],split[i-1],split[i],colors= 'orange',linestyles='dashed',label='Optimal WOB')
    ax[4].plot(time_dict,depth_dict)
    ax[4].set_ylabel('Depth[ft/hr]')
    ax[5].plot(time_dict, drillability)
    ax[5].set_ylabel('Drillability constant[0,1]')
    ax[5].set_xlabel('Time[hr]')
    plt.show()

    driller_plot_rop = np.flip(rop_dict)
    driller_plot_depth = np.flip(depth_dict)
    plt.figure('driller plot')
    plt.plot(driller_plot_rop, driller_plot_depth)
    plt.gca().invert_yaxis()
    plt.ylabel('Depth[ft]')
    plt.xlabel('ROP[ft/hr]')
    plt.show()
