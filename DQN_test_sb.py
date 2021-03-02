import gym
import rop_envs

from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

###NB Optimal ROP i denne casen: 117.61340377001153
###Optimal WOB = 36

env = gym.make('eckel-disc-v0')


model = DQN(MlpPolicy, env, verbose = 1)
model.learn(total_timesteps = 10000, log_interval = 1)
model.save("dqn_eckel_mono")
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(mean_reward)
print(std_reward)