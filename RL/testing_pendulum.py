import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make('CartPole-v1')
model = PPO(MlpPolicy, env, verbose = 0)

model.learn(total_timesteps=1000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")





