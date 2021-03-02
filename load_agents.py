import gym
import rop_envs

from stable_baselines3 import A2C, DDPG
from stable_baselines3.common.evaluation import evaluate_policy

ekcel_multidiscrete = gym.make('eckel-mdisc-v0')

actor_critic = A2C.load('trained_agents\\a2c_mdisc_108_485')
observation = ekcel_multidiscrete.reset()
action_list = []
for i in range(0,100):
    action, _states = actor_critic.predict(observation)
    observation, reward, done, info = ekcel_multidiscrete.step(action)
    action_list.append(action)
    ekcel_multidiscrete.render()
print('actions taken by actor critc agent: ' , action_list)


