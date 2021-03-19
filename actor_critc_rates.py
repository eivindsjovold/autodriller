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
markenv = gym.make('bm-v0')
case = 'load'
savestring = 'trained_agents\\vary_hardness_a2c_eckel_mod_reward'
loadstring = 'trained_agents\\vary_hardness_a2c_eckel_mod_reward'
states = []
depth = []
rop = []

if case == 'train':
    model = A2C(MlpPolicy, vec_env, verbose = 1)
    model.learn(total_timesteps = 5000000)
    model.save(savestring)

elif case == 'train_more':
    model = A2C.load(loadstring)
    model.set_env(vec_env)
    model.learn(total_timesteps=10000000)
    model.save(savestring)

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

elif case == 'benchmark':
    K = 0
    K_vec =[0.15, 0.234, 0.13, 0.3, 0.1, 0.1778]
    model = A2C.load(loadstring)
    obs = markenv.reset(K)
    hard = []
    mark_len=[]
    counter = 0
    for i in range(len(K_vec)):
        done = False
        obs = markenv.reset(K_vec[i])
        print('iteration',i)
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = markenv.step(action)
            #markenv.render()
            states.append(obs)
            rop.append(info[0])
            depth.append(obs[3])
            counter += 1
        hard.append(info[1])
        mark_len.append(counter)
    plt.figure()
    plt.plot(rop, label = 'ROP')
    plt.plot([0,mark_len[0]],[202.63960366840615,202.63960366840615], label = 'optimum for hardness_1')
    plt.plot([mark_len[0],mark_len[1]],[384.2365391173962,384.2365391173962], label = 'optimum for hardness_2')
    plt.plot([mark_len[1],mark_len[2]],[164.92799124975556,164.92799124975556], label = 'optimum for hardness_3')
    plt.plot([mark_len[2],mark_len[3]],[549.3352038475479,549.3352038475479], label = 'optimum for hardness_4')
    plt.plot([mark_len[3],mark_len[4]],[113.06778714097555,113.06778714097555], label = 'optimum for hardness_5')
    plt.plot([mark_len[4],mark_len[5]],[258.80306473127195,258.80306473127195], label = 'optimum for hardness_6')
    plt.legend(loc = 'upper left')
    plt.xlabel('Time[s]')
    plt.ylabel('ROP[ft/hr]')
    plt.title('RL agent inspired by Extremum Seeking Policy')
    plt.savefig('vary_hardness_eckel')
    plt.show()
        
    

else:
    print('no valid case specified.')    