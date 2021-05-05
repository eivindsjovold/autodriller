import os
import gym
import rop_envs
import csv
from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy as mlp
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import tensorboard
# Prøv å trene på lave verdier

env = DummyVecEnv([lambda: gym.make('memory-eckel-v0')])
env = VecNormalize(env)

vec_env = make_vec_env('memory-eckel-v0', n_envs=16)
vec_env = VecNormalize(vec_env)
##
env = vec_env
##
case = 'load'
savestring = 'trained_agents\\memory_eckel_5_may_normalization'
loadstring = 'trained_agents\\memory_eckel_5_may_normalization'
stat_logger = 'statistics_normalization/eckel/'
log_name = 'eckel1.pkl'

stats_path = os.path.join(stat_logger,log_name)

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
counter = 0

if case == 'train':
    print(case)
    model = A2C(MlpPolicy, env, verbose = 1, policy_kwargs=dict(optimizer_class=RMSpropTFLike, optimizer_kwargs = dict(alpha = 0.99, eps = 1e-5, weight_decay = 0)), tensorboard_log='./tensorboard_logs')
    model.learn(total_timesteps = 2500000, tb_log_name='memory_eckel')
    model.save(savestring)
    env.save(stats_path)

elif case == 'train_more':
    print(case)
    model = A2C.load(loadstring)
    model.set_env(env)
    model.learn(total_timesteps=250000)
    model.save(savestring)

elif case == 'load':
    print(stats_path)
    model = A2C.load(loadstring)
    env = DummyVecEnv([lambda: gym.make('memory-eckel-v0')])
    env = VecNormalize.load(stats_path, env)
    env.training = False
    env.norm_reward = False
    normalized_observations = []
    ## Actual interaction
    for i in range(2):
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)   
            normalized_observations.append(obs)
            reward.append(rewards)            
    ## Process data
    counter = 0
    depth = 0
    for i in range(len(normalized_observations)-1):    
        obs = env.unnormalize_obs(normalized_observations[i])
        counter += 1
        wob = obs[0][0]
        rpm = obs[0][2]
        q = obs[0][4]
        rop = obs[0][6]
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
    fig, ax = plt.subplots(5)
    fig.suptitle('RL agent inspired by extremum seeking')
    ax[0].plot(rop_dict)
    ax[0].plot([0,iteration[-1]],[212.82898999287448,212.82898999287448])
    ax[0].set_title('ROP[ft/hr]')
    ax[1].plot(wob_dict)
    ax[1].plot([0,iteration[-1]],[50,50])
    ax[2].plot(rpm_dict)
    ax[2].plot([0,iteration[-1]],[115,115])
    ax[3].plot(q_dict)
    ax[3].plot([0,iteration[-1]],[103,103])
    ax[4].plot(reward)
    
    #plt.savefig('figures\\BY_load_agent')
    plt.show()
