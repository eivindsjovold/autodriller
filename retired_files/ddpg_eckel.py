from stable_baselines3.common.env_checker import check_env

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.evaluation import evaluate_policy

import gym
import rop_envs
case = 'train_more'
env1 = gym.make('eckel-cont-v0')
if case == 'train':
    n_actions = env1.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG('MlpPolicy', env1, action_noise=action_noise, verbose=1)

    model.learn(total_timesteps = 2000, log_interval=1)

    #mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)
    #print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    model.save("test-DDPG")

elif case == 'train_more':
    n_actions = env1.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma = 0.1 * np.ones(n_actions))
    model = DDPG.load('test-DDPG')
    model.set_env(env1)
    model.learn(total_timesteps=100000, log_interval = 10)
    model.save('trained_agents\\test-DDPG')
