import gym
import rop_envs
from stable_baselines3 import A2C, PPO
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env
import numpy as np

env = gym.make('simple-v1')
vec_env = make_vec_env('simple-v1', n_envs = 8)
case = 'load'

savestring = 'trained_agents\\test_simple_optimum_vary_vec'
loadstring = 'trained_agents\\test_simple_optimum_vary'

if case == 'train':
    model = A2C(MlpPolicy, env, verbose = 1)
    model.learn(total_timesteps=100000)
    model.save(savestring)

elif case == 'load':
    model = PPO.load(loadstring)
    obs = env.reset()
    done = False
    rop = []
    wob = []
    optimal = []
    counter = 0
    count = []
    iteration = []
    rewards = []
    for i in range(2):
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            #env.render()
            rop.append(obs[2])
            wob.append(obs[0])
            rewards.append(reward)
            counter += 1
            iteration.append(counter)
        optimal.append(info)
        count.append(counter)
    plt.subplot(311)
    plt.plot(rop)
    plt.subplot(312)
    plt.plot(wob, color ='blue', linestyle = 'dashed')
    plt.hlines(optimal[0],0,count[0], colors = 'orange', linestyles = '--')
    plt.hlines(optimal[1],count[0],count[1], colors = 'orange', linestyles = '--')
    plt.subplot(313)
    plt.scatter(iteration, rewards)
    plt.show()

elif case == 'train_more':
    model = A2C.load(loadstring, verbose = 1)
    model.set_env(env)
    model.learn(total_timesteps=100000)
    model.save(savestring)

elif case == 'ppo':
    model = PPO.load('trained_agents\\ppo_simple_env')
    model.set_env(env)
    model.learn(total_timesteps = 250000)
    model.save('trained_agents\\ppo_simple_env')
     