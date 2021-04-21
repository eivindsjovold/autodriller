import gym
import rop_envs
from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env

env = gym.make('by-v1')
vec_env = make_vec_env('by-v1', n_envs=30)

case = 'load'
savestring = 'trained_agents\\by_for_thesis'
loadstring = 'agents_thesis\\by_for_thesis'
#check_env(env)

wob_dict = []
rpm_dict = []
flow_dict = []
depth_dict = []
rop_dict = []
iteration = []
reward = []
split = []
states = []

if case == 'train':
    model = A2C(MlpPolicy, vec_env, verbose = 1)
    model.learn(total_timesteps = 1000000)
    model.save(savestring)

elif case == 'train_more':
    for i in range(10):
        print('training session:',i)    
        model = A2C.load(loadstring, verbose = 1)
        model.set_env(vec_env)
        model.learn(total_timesteps=1000000)
        model.save(savestring)

elif case =='load':
    model = A2C.load(loadstring)
    obs = env.reset()
    counter = 0
    for i in range(3):
        done = False
        obs = env.reset()
        while not done:
            counter += 1
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            env.render()
            wob_dict.append(obs[0])
            rpm_dict.append(obs[1])
            flow_dict.append(obs[2])
            rop_dict.append(obs[3])
            reward.append(rewards)
    fig, ax = plt.subplots(4)
    fig.suptitle('RL agent inspired by extremum seeking')
    ax[0].plot(rop_dict)
    ax[0].set_title('ROP[ft/hr]')
    ax[1].plot(wob_dict, label = 'WOB')
    ax[1].set_title('WOB[klbf]')
    ax[2].plot(rpm_dict)
    ax[2].set_title('RPM[rev/min]')
    ax[3].plot(flow_dict)
    ax[3].set_title('flow')
    #ax[4].plot(reward)
    #ax[4].set_title('reward')
    #plt.savefig('figures\\BY_load_agent')
    plt.show()


        
    

else:
    print('no valid case specified.')    
    