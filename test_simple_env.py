import gym
import rop_envs
from stable_baselines3 import A2C, PPO
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env
from models.simple_model import rop_multi
import numpy as np
import tensorflow
import tensorboard

env = gym.make('simple-v2')
vec_env = make_vec_env('simple-v1', n_envs = 8)
case = 'multivariate'

savestring = 'trained_agents\\test_simple_optimum_multivariate_multiplicate'
loadstring = 'trained_agents\\test_simple_optimum_multivariate'

if case == 'train':
    model = A2C(MlpPolicy, env, verbose = 1, tensorboard_log='./tensorboard_logs')
    model.learn(total_timesteps=250000, tb_log_name='simple_model_multivariate')
    model.save(savestring)

elif case == 'load':
    model = A2C.load(loadstring)
    obs = env.reset()
    done = False
    rop = []
    wob = []
    optimal = []
    counter = 0
    count = []
    iteration = []
    rewards = []
    for i in range(3):
        print('iteration', i)
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            #env.render()
            rop.append(obs[2])
            wob.append(obs[0])
            rewards.append(reward)
            counter += 1
            iteration.append(counter)
        #optimal.append(info)
        count.append(counter)
    plt.subplot(311)
    plt.plot(rop)
    plt.hlines(optimal[0], 0, count[0], colors = 'red', linestyles=  '--')
    plt.subplot(312)
    plt.plot(wob, color ='blue', linestyle = 'dashed')
    plt.hlines(optimal[0],0,count[0], colors = 'red', linestyles = '--')
    plt.hlines(optimal[1],count[0],count[1], colors = 'orange', linestyles = '--')
    plt.subplot(313)
    plt.scatter(iteration, rewards)
    plt.show()
    
    '''
    plt.figure()
    plt.plot([0,count[0]],[optimal[0], optimal[0]], color = 'red', zorder = 1, linewidth = 3, label = 'Optimal Value')
    plt.plot(wob, zorder = 100, linestyle = 'dotted', label = 'WOB')
    plt.xlabel('Iterations')
    plt.ylabel('WOB')
    plt.legend()
    #plt.savefig('thesis_fig\\simple_opt_wob_25.eps', format = 'eps')
    #plt.show()

    plt.figure()
    plt.plot([0,count[0]],[optimal[0], optimal[0]], color = 'red', zorder = 1, label = 'Optimal Value')
    plt.plot(rop, zorder = 2, linestyle = 'dotted', label = 'ROP')
    plt.xlabel('Iterations')
    plt.ylabel('ROP')
    plt.legend()
    #plt.savefig('thesis_fig\\simple_opt_rop_25.eps', format = 'eps')
    '''
    plt.show()

elif case == 'train_more':
    model = A2C.load(loadstring, verbose = 1)
    model.set_env(env)
    model.learn(total_timesteps=100000)
    model.save(savestring)

elif case == 'ppo':
    model = PPO.load('trained_agents\\ppo_simple_env')
    model.set_env(env)
    model.learn(total_timesteps = 250000)
    model.save('trained_agents\\ppo_simple_env')

elif case == 'multivariate':
    print(case)
    model = A2C.load(loadstring)
    obs = env.reset()
    done = False
    rop = []
    wob = []
    q = []
    rpm = []
    optimal = []
    counter = 0
    count = []
    iteration = []
    rewards = []
    for i in range(3):
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            #env.render()
            rop.append(obs[6])
            wob.append(obs[0])
            q.append(obs[4])
            rpm.append(obs[2])
            rewards.append(reward)
            counter += 1
            iteration.append(counter)
        optimal.append(info)
        count.append(counter)
    plt.subplot(511)
    plt.plot(rop)
    #plt.hlines(optimal[0], 0, count[0], colors = 'red', linestyles=  '--')
    plt.subplot(512)
    plt.plot(wob, color ='blue', linestyle = 'dashed')
    plt.plot([0,iteration[-1]],[optimal[0][0],optimal[0][0]])
    #plt.hlines(optimal[0],0,count[0], colors = 'red', linestyles = '--')
    #plt.hlines(optimal[1],count[0],count[1], colors = 'orange', linestyles = '--')
    plt.subplot(513)
    plt.plot(q,color ='blue', linestyle = 'dashed')
    plt.plot([0,iteration[-1]],[optimal[0][2],optimal[0][2]])
    plt.subplot(514)
    plt.plot(rpm,color ='blue', linestyle = 'dashed')
    plt.plot([0,iteration[-1]],[optimal[0][1],optimal[0][1]])
    plt.subplot(515)
    plt.scatter(iteration, rewards)
    
    rop_opt = rop_multi(optimal[0][0],optimal[0][1],optimal[0][2],optimal[0][0],optimal[0][1],optimal[0][2])
    plt.figure()
    plt.plot([0,count[0]],[optimal[0][0], optimal[0][0]], color = 'red', zorder = 1, linewidth = 3, label = 'Optimal Value')
    plt.plot(wob, zorder = 100, linestyle = 'dotted', label = 'WOB')
    plt.xlabel('Iterations')
    plt.ylabel('WOB')
    plt.legend()
    plt.savefig('thesis_fig\\simple_multi\\1.5_simple_opt_wob_multi.eps', format = 'eps')
    #plt.show()

    plt.figure()
    plt.plot([0,count[0]],[rop_opt, rop_opt], color = 'red', zorder = 1,  linewidth = 3,label = 'Optimal Value')
    plt.plot(rop, zorder = 2, linestyle = 'dotted', label = 'ROP')
    plt.xlabel('Iterations')
    plt.ylabel('ROP')
    plt.legend()
    plt.savefig('thesis_fig\\simple_multi\\1.5_simple_opt_rop_multi.eps', format = 'eps')

    plt.figure()
    plt.plot([0,count[0]],[optimal[0][1], optimal[0][1]], color = 'red', zorder = 1, linewidth = 3, label = 'Optimal Value')
    plt.plot(rpm, zorder = 2, linestyle = 'dotted', label = 'RPM')
    plt.xlabel('Iterations')
    plt.ylabel('RPM')
    plt.legend()
    plt.savefig('thesis_fig\\simple_multi\\1.5_simple_opt_rpm_multi.eps', format = 'eps')
    
    plt.figure()
    plt.plot([0,count[0]],[optimal[0][2], optimal[0][2]], color = 'red', zorder = 1, linewidth = 3, label = 'Optimal Value')
    plt.plot(q, zorder = 2, linestyle = 'dotted', label = 'Q')
    plt.xlabel('Iterations')
    plt.ylabel('Q')
    plt.legend()
    plt.savefig('thesis_fig\\simple_multi\\1.5_simple_opt_q_multi.eps', format = 'eps')
    
    
    
      
    
    plt.show()