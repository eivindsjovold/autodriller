from stable_baselines3.common.env_checker import check_env

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.evaluation import evaluate_policy

import gym
import testenv
import rop_envs

env = gym.make('rop-v0')
n_actions = env.action_space.shape[-1]

action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG('MlpPolicy', env, action_noise=action_noise, verbose=1)

model.learn(total_timesteps = 100000, log_interval=1)

#mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)
#print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
model.save("test-DDPG")
