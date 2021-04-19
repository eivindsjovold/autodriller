import gym
import rop_envs
from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt


env = gym.make('rop-v0')
vec_env = make_vec_env('rop-v0', n_envs=30)
markenv = gym.make('bm-v0')

case = 'train_more'
savestring = 'trained_agents\\for_thesis_eckel'
loadstring = 'trained_agents\\for_thesis_eckel'

wob = []
rpm = []
q = []
depth = []
rop = []
iteration = []
reward = []
split = []
states = []

if case == 'train':
    model = A2C(MlpPolicy, vec_env, verbose = 1)
    model.learn(total_timesteps = 1000000)
    model.save(savestring)


elif case =='load':
    model = A2C.load(loadstring)
    counter = 0
    for i in range(0,10):
        done = False
        obs = env.reset()
        while not done:
            counter += 1
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            #env.render()
            wob.append(obs[0])
            rpm.append(obs[1])
            depth.append(obs[2])
            reward.append(rewards)
            iteration.append(counter)
            rop.append(info)
    print(split)
    fig, ax = plt.subplots(5)
    fig.suptitle('RL agent inspired by extremum seeking')
    ax[0].plot(rop)
    ax[0].set_title('ROP[ft/hr]')
    
    plt.savefig('figures\\BY_load_agent')
    plt.show()

elif case == 'train_more':
    for i in range(10):
        print('training session:',i)    
        model = A2C.load(loadstring, verbose = 1)
        model.set_env(vec_env)
        model.learn(total_timesteps=1000000)
        model.save(savestring)