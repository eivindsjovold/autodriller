import gym
import rop_envs

from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make('eckel-v0')
model = DQN.load('trained_agents\\dqn_eckel_mono_116.52')
obs = env.reset()

for i in range(0,1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()