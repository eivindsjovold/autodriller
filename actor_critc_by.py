import gym
import rop_envs
from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env
import csv
import numpy as np

env = gym.make('by-v1')
vec_env = make_vec_env('by-v1', n_envs=30)
test_env = gym.make('by-test-v1')

case = 'load'
savestring = 'trained_agents\\by_for_thesis_26_april_reduced_reward_changed_wob_founder'
loadstring = 'trained_agents\\by_for_thesis_26_april_reduced_reward_changed_wob_founder'
#check_env(env)

wob_dict = []
rpm_dict = []
flow_dict = []
depth_dict = []
rop_dict = []
iteration = []
time_dict = []
reward = []
split = []
states = []

if case == 'train':
    model = A2C(MlpPolicy, vec_env, verbose = 1)
    model.learn(total_timesteps = 1000000)
    model.save(savestring)

elif case == 'train_more':
    #for i in range(10):
        #print('training session:',i)    
    model = A2C.load(loadstring, verbose = 1)
    model.set_env(vec_env)
    model.learn(total_timesteps=5000000)
    model.save(savestring)

elif case =='load':
    model = A2C.load(loadstring)
    obs = env.reset()
    counter = 0
    for i in range(10):
        done = False
        obs = env.reset()
        while not done:
            counter += 1
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            #env.render()
            wob_dict.append(obs[0])
            rpm_dict.append(obs[1])
            flow_dict.append(obs[2])
            rop_dict.append(obs[3])
            reward.append(rewards)
    fig, ax = plt.subplots(4)
    fig.suptitle('RL agent inspired by extremum seeking')
    ax[0].plot(rop_dict)
    ax[0].set_title('ROP[ft/hr]')
    ax[1].plot(wob_dict, label = 'WOB')
    ax[1].set_title('WOB[klbf]')
    ax[2].plot(rpm_dict)
    ax[2].set_title('RPM[rev/min]')
    ax[3].plot(flow_dict)
    ax[3].set_title('flow')
    #ax[4].plot(reward)
    #ax[4].set_title('reward')
    #plt.savefig('figures\\BY_load_agent')
    plt.show()


elif case == 'test_byenv':

    opt =[]
    wob_opt = []
    rpm_opt = []
    flow_opt = []
    rop_opt = []
    a1 = []
    a5 = []
    a6 = []
    a8 = []
    length = []
    with open('parameters_for_BY.csv', mode = 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter = ',')
        for row in csv_reader:
            a1.append(float(row[0]))
            a5.append(float(row[1]))
            a6.append(float(row[2]))
            a8.append(float(row[3]))
            length.append(int(row[4]))
        
        csvfile.close()
    
    model = A2C.load(loadstring)
    counter = 0
    wob = 10
    rpm = 10
    q = 10
    rop = 0
    split = []
    depth = 0
    time = 0
    delta_t = 1/3600
    init_vals = np.array([50,50,50,150],dtype = np.float32)
    for i in range(len(a1)):
        done = False
        obs = test_env.reset(a1[i],length[i],init_vals)
        while not done:
            counter += 1
            time += 1/3600
            action, _states = model.predict(obs)
            obs, rewards, done, info = test_env.step(action)

            #if counter % 200 == 0:
            test_env.render()
            wob = obs[0]
            rpm = obs[1]
            q = obs[2]
            rop = obs[3]
            depth += rop*delta_t
            rop_dict.append(rop)
            time_dict.append(time)
            wob_dict.append(wob)
            rpm_dict.append(rpm)
            flow_dict.append(q)
            depth_dict.append(depth)
            reward.append(rewards)
        split.append(time)
        init_vals = np.array([wob,rpm,q,rop], dtype = np.float32)

    with open('optimal_value_BY.csv', mode = 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter = ',')
        for row in csv_reader:
            wob_opt.append(float(row[4]))
            rpm_opt.append(float(row[5]))
            flow_opt.append(float(row[6]))
            rop_opt.append(float(row[7]))        
        csvfile.close()       

    fig, ax = plt.subplots(5)
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
    ax[3].plot(time_dict,flow_dict)
    ax[3].set_ylabel('Q[gal/min]')
    ax[3].hlines(flow_opt[0],0,split[0],colors= 'orange',linestyles='dashed',label='Optimal q')
    for i in range(1,len(split)):
        ax[3].hlines(flow_opt[i],split[i-1],split[i],colors= 'orange',linestyles='dashed',label='Optimal WOB')
    ax[4].plot(time_dict,depth_dict)
    ax[4].set_ylabel('Depth[ft/hr]')
    ax[4].set_xlabel('Time[hr]')
    plt.show()

    driller_plot_rop = np.flip(rop_dict)
    driller_plot_depth = np.flip(depth_dict)
    plt.figure('driller plot')
    plt.plot(driller_plot_rop, driller_plot_depth)
    plt.gca().invert_yaxis()
    plt.ylabel('Depth[ft]')
    plt.xlabel('ROP[ft/hr]')
    plt.show()



        
    

else:
    print('no valid case specified.')    
    