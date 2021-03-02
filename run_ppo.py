import gym
import rop_envs

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

env = make_vec_env('by-v0', n_envs = 4)
model = PPO(MlpPolicy, env, verbose = 1)
model.learn(total_timesteps=100000)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
