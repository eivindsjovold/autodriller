import gym
import rop_envs
from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt

env = gym.make('rop-v0')
by_test_env = gym.make('rop-v1')
vec_env = make_vec_env('rop-v0', n_envs=10)
case = 'train'
states = []

if case == 'train':
    model = A2C(MlpPolicy, by_test_env, verbose = 1)
    model.learn(total_timesteps = 10000)
    model.save('trained_agents\\by_a2c_rates')
elif case == 'train_more':
    model = A2C.load('trained_agents\\a2c_rates')
    model.set_env(vec_env)
    model.learn(total_timesteps=500000)
elif case =='load':
    model = A2C.load('trained_agents\\a2c_rates')
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        states.append(obs)
        env.render()
        if dones:
            break

else:
    print('no valid case specified.')    