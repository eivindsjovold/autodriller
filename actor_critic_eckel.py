import gym
import rop_envs
from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
#import tensorboard



env = gym.make('eckel-v0')
vec_env = make_vec_env('eckel-v0', n_envs=30)
env2 = gym.make('eckel-v1')
vec_env2 = make_vec_env('eckel-v1', n_envs = 30)

markenv = gym.make('bm-v0')
test_env = gym.make('rop-iv-v0')
case = 'train'
savestring = 'trained_agents\\for_thesis_eckel'
loadstring = 'trained_agents\\for_thesis_eckel'

wob_dict = []
rpm_dict = []
q_dict = []
iteration = []
reward = []
split = []
time_dict = []
rop_dict = []
depth_dict = []
time = 0
flow = []
delta_t = 1/3600

if case == 'train':
    model = A2C(MlpPolicy, vec_env2, verbose = 1)#, tensorboard_log='./tensorboard_logs')
    model.learn(total_timesteps = 2000000)#, tb_log_name='session_eckelV2')
    model.save(savestring)


elif case =='load':
    model = A2C.load(loadstring)
    counter = 0
    wob = 10
    rpm = 10
    q = 10
    rop = 0
    depth = 0
    for i in range(0,10):
        done = False
        obs = env2.reset()
        obs = np.array([wob,rpm,q,rop], dtype = np.float32)
        while not done:
            counter += 1
            time += 1/3600
            action, _states = model.predict(obs)
            obs, rewards, done, info = env2.step(action)
            wob = obs[0]
            rpm = obs[1]
            q = obs[2]
            rop = obs[3]
            depth += rop*delta_t
            rop_dict.append(rop)
            time_dict.append(time)
            wob_dict.append(wob)
            rpm_dict.append(rpm)
            q_dict.append(q)
            depth_dict.append(depth)
            reward.append(rewards)
    fig, ax = plt.subplots(5)
    fig.suptitle('RL agent inspired by extremum seeking')
    ax[0].plot(time_dict, rop_dict)
    ax[0].set_title('ROP[ft/hr]')
    ax[1].plot(time_dict, wob_dict)
    ax[2].plot(time_dict, rpm_dict)
    ax[3].plot(time_dict,q_dict)
    
    #plt.savefig('figures\\BY_load_agent')
    plt.show()

elif case == 'train_more':
    for i in range(10):
        print('training session:',i)    
        model = A2C.load(loadstring, verbose = 1)
        model.set_env(vec_env2)
        model.learn(total_timesteps=1000000)
        model.save(savestring)