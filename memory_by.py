import os
import gym
import rop_envs
import csv
from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy, CnnPolicy
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy as mlp
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import tensorboard
# Prøv å trene på lave verdier

env = DummyVecEnv([lambda: gym.make('memory-by-v0')])
env = VecNormalize(env)

vec_env = make_vec_env('memory-by-v0', n_envs=16)
vec_env = VecNormalize(vec_env)

case = 'load'
stat_logdir = 'statistics_normalization/'
agent_logdir = 'trained_agents/'


savestring = agent_logdir + 'memory_by_5_may_normalization'
loadstring = agent_logdir + 'memory_by_5_may_normalization'
stats_path = os.path.join(stat_logdir, 'bymemory1.pkl')

env = vec_env

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
    model = A2C(MlpPolicy, env, verbose = 1, policy_kwargs=dict(optimizer_class=RMSpropTFLike, optimizer_kwargs = dict(alpha = 0.99, eps = 1e-5, weight_decay = 0)), tensorboard_log='./tensorboard_logs')
    model.learn(total_timesteps = 3000000, tb_log_name='memory_eckel')
    model.save(savestring)
    vec_env.save(stats_path)

elif case == 'train_more':
    print(case)
    model = A2C.load(loadstring)#,  policy_kwargs=dict(optimizer_class=RMSpropTFLike, optimizer_kwargs = dict(alpha = 0.99, eps = 1e-5, weight_decay = 0)))
    env = DummyVecEnv([lambda: gym.make('memory-by-v0')])
    env = VecNormalize.load(stats_path, env)
    model.set_env(env)
    model.learn(total_timesteps=500000)
    model.save(savestring)
    env.save(stats_path)

elif case == 'load':
    print(stats_path)
    print(loadstring)
    model = A2C.load(loadstring)
    env = DummyVecEnv([lambda: gym.make('memory-by-v0')])
    env = VecNormalize.load(stats_path, env)
    env.training = True
    env.norm_reward = False
    counter = 0
    depth = 0
    normalized_observations = []
    # Loop through normalized agent
    for i in range(1):
        done = False
        obs = env.reset()
        #print(obs)
        while not done:
            counter += 1
            time += 1/3600
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)   
            normalized_observations.append(obs)
            reward.append(rewards)
    #unnormalize observations and plot. Might be some errors in running averages
    for i in range(len(normalized_observations)-1):
        obs = env.unnormalize_obs(normalized_observations[i])
        wob = obs[0][0]
        rpm = obs[0][4]
        q = obs[0][8]
        rop = obs[0][12]
        depth += rop*delta_t
        rop_dict.append(rop)
        time_dict.append(time)
        wob_dict.append(wob)
        rpm_dict.append(rpm)
        q_dict.append(q)
        depth_dict.append(depth)
        iteration.append(counter)
    fig, ax = plt.subplots(5)
    fig.suptitle('RL agent inspired by extremum seeking')
    ax[0].plot(rop_dict)
    ax[0].plot([0,iteration[-1]],[339.885475592095,339.8854755920957])
    ax[0].set_title('ROP[ft/hr]')
    ax[1].plot(wob_dict)
    ax[1].plot([0,iteration[-1]],[111,111])
    ax[2].plot(rpm_dict)
    ax[2].plot([0,iteration[-1]],[214,214])
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
        env.render()
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
