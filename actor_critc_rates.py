import gym
import rop_envs
from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt

env = gym.make('rop-v0')
by_env = gym.make('rop-v1')
vec_env = make_vec_env('rop-v0', n_envs=10)
by_vec_env = make_vec_env('rop-v1', n_envs = 10)
case = 'load'
savestring = 'trained_agents\\vary_hardness_a2c_eckel'
loadstring = 'trained_agents\\vary_hardness_a2c_eckel'
states = []
depth = []
rop = []

if case == 'train':
    model = A2C(MlpPolicy, vec_env, verbose = 1)
    model.learn(total_timesteps = 400000)
    model.save(savestring)

elif case == 'train_more':
    model = A2C.load(loadstring)
    model.set_env(vec_env)
    model.learn(total_timesteps=2500000)

elif case =='load':
    model = A2C.load(loadstring)
    obs = env.reset()
    for i in range(10):
        done = False
        obs = env.reset()
        print('iteration',i)
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            env.render()
            states.append(obs)
            rop.append(info)
            depth.append(obs[3])
        
    plt.figure()
    plt.plot(rop, label = 'ROP')
    plt.xlabel('Time[s]')
    plt.ylabel('ROP[ft/hr]')
    plt.title('RL agent inspired by Extremum Seeking Policy')
    plt.savefig('vary_hardness_eckel')
    plt.show()
    

else:
    print('no valid case specified.')    