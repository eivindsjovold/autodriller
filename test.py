import gym
import rop_envs
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_checker import check_env

env = gym.make('rop-v0')
model = PPO(MlpPolicy, env, verbose = 1)
model.learn(total_timesteps = 100000)