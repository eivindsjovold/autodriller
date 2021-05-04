import gym
import rop_envs
import csv
from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy as mlp
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
#import tensorflow as tf
import numpy as np
#import tensorboard
# Prøv å trene på lave verdier

env = gym.make('memory-by-v0')
vec_env = make_vec_env('memory-by-v0', n_envs=16)


case = 'load'
savestring = 'trained_agents\\memory_by_1_may_build'
loadstring = 'trained_agents\\memory_by_1_may_build'


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
    print(case)
    model = A2C(MlpPolicy, vec_env, verbose = 1)#, tensorboard_log='./tensorboard_logs')
    model.learn(total_timesteps = 2500000)#, tb_log_name='memory_eckel')
    model.save(savestring)

elif case == 'train_more':
    print(case)
    model = A2C.load(loadstring)
    model.set_env(vec_env)
    model.learn(total_timesteps=1000000)
    model.save(savestring)

elif case == 'load':
    model = A2C.load(loadstring)
    counter = 0
    depth = 0
    for i in range(1):
        done = False
        obs = env.reset()
        while not done:
            counter += 1
            time += 1/3600
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            env.render()
            wob = obs[0]
            rpm = obs[2]
            q = obs[4]
            rop = obs[6]
            depth += rop*delta_t
            rop_dict.append(rop)
            time_dict.append(time)
            wob_dict.append(wob)
            rpm_dict.append(rpm)
            q_dict.append(q)
            depth_dict.append(depth)
            reward.append(rewards)
            iteration.append(counter)
    print(counter)
    print(sum(reward))
    fig, ax = plt.subplots(5)
    fig.suptitle('RL agent inspired by extremum seeking')
    ax[0].plot(iteration, rop_dict)
    ax[0].plot([0,iteration[-1]],[312.7068437460617,312.7068437460617])
    ax[0].set_title('ROP[ft/hr]')
    ax[1].plot(wob_dict)
    ax[1].plot([0,iteration[-1]],[111,111])
    ax[2].plot(rpm_dict)
    ax[2].plot([0,iteration[-1]],[204,204])
    ax[3].plot(q_dict)
    ax[3].plot([0,iteration[-1]],[87,87])
    ax[4].plot(reward)
    
    #plt.savefig('figures\\BY_load_agent')
    plt.show()

elif case == 'indi':
    model = A2C.load('trained_agents\\test_simple_optimum')
    env = gym.make('wob-v0')
    obs = env.reset()
    done = False
    rop = []
    wob = []
    rewards = []
    counter = 0
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        #env.render()
        rop.append(obs[2])
        wob.append(obs[0])
        rewards.append(reward)
        counter += 1
        iteration.append(counter)
    #optimal.append(info)
    count.append(counter)
    plt.subplot(311)
    plt.plot(rop)
    #plt.hlines(optimal[0], 0, count[0], colors = 'red', linestyles=  '--')
    plt.subplot(312)
    plt.plot(wob, color ='blue', linestyle = 'dashed')
    plt.hlines(optimal[0],0,count[0], colors = 'red', linestyles = '--')
    #plt.hlines(optimal[1],count[0],count[1], colors = 'orange', linestyles = '--')
    plt.subplot(313)
    plt.scatter(iteration, rewards)
    plt.show()
