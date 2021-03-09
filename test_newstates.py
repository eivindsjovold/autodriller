import gym
import rop_envs
from stable_baselines3 import PPO, DQN
from stable_baselines3.dqn import MlpPolicy as mldq
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt

###########
case = 'load'
model_name = 'trained_agents\\test_rates_ppo'
env = gym.make('rop-v0')


##########
if case == 'train':
    model = PPO(MlpPolicy, env, verbose = 1)
    model.learn(total_timesteps=500000)

    model.save('trained_agents\\test_rates_ppo')
elif case == 'load':
    model = PPO.load('trained_agents\\test_rates_ppo')
    obs = env.reset()
    states = []
    rop = []
    iteration = 0
    vec = []
    while True:
        iteration += 1
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        states.append(obs)
        #env.render()
        if dones:
            break
    
elif case == 'train_more':
    model = PPO.load(model_name)
    model.set_env(env)
    model.learn(total_timesteps=100000) 
else: 
    print('no such case defined')