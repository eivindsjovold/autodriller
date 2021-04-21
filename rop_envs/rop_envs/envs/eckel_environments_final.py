import numpy as np
import gym
import random
from gym import spaces

from models.eckels.eckel import rate_of_penetration_eckel
from models.eckels.eckel import rate_of_penetration_eckel_individual_founder, rate_of_penetration_eckel_vary_founder

class EckelEnv1(gym.Env):
    plots = []
    metadata = {'render.modes': ['human', 'ansi']}
    mode = 'human'
    MAX_WOB = 200
    MAX_RPM = 300
    MAX_Q = 400
    depth_final = random.uniform(50, 250)
    delta_t = 1 /3600
    v = 0.1  #feet/s
    a11 = 0.005
    a22 = 0.005
    a33 = 0.005
    a = 1
    b = 0.75
    c = 0.5
    k = 0.1
    rho = 1000
    d_n = 0.25
    my = 0.4
    ub_k = 0.9
    lb_k = 0.025


    def __init__(self):
        self.viewer = True
        self.K = random.uniform(self.lb_k, self.ub_k)
        self.state = np.array([10,10,10,0], dtype = np.float32)
        self.reward = 0
        self.last_rop = 0
        self.counter = 0
        high = np.array([self.MAX_WOB, self.MAX_RPM, self.MAX_Q, 100000], dtype = np.float32)
        low = np.array([0,0,0,0], dtype = np.float32)
        self.observation_space = spaces.Box(low = low, high = high, dtype = np.float32)
        #down, stay, up
        self.action_space = spaces.MultiDiscrete([3,3,3])


    
    def actionToValue(self, action):
        rates = np.zeros(3)
        for i in range(0, len(action)):
            if action[i] == 0:
                rates[i] = -1
            elif action[i] == 1:
                rates[i] = 0
            elif action[i] == 2:
                rates[i] = 1
        return rates
    
    def isDone(self):
        return not self.depth < self.depth_final

    def step(self, action):
        wob = self.state[0]
        rpm = self.state[1]
        q = self.state[2]
        rop = self.state[3]
        rates = self.actionToValue(action)
        reward, rop = self.alternative_reward(rates)
        self.state[0] += rates[0]
        self.state[1] += rates[1]
        self.state[2] += rates[2]
        self.state[3] = rop
        self.counter = self.useless_counter(rop,self.counter)
        done = self.isDone()
        if done:
            reward = + 1000
        if self.counter >= 10:
            reward = -1000
            done = True
        self.reward = reward
        return self.state, reward, done, {}

    #def change_formation(self):
    #    self.K = random.uniform(self.lb_k, self.ub_k)
    #    self.formation_depth = 0
    #    self.disturb = random.uniform(self.state[3] + 5, self.state[3] + 100 )



    def check_for_nan(self,number):
        return number != number

    def alternative_reward(self, rates):
        wob = self.state[0] + rates[0]
        rpm = self.state[1] + rates[1]
        q = self.state[2] + rates[2]
        depth = self.state[3]
        rop = rate_of_penetration_eckel_individual_founder(self.a, self.b, self.c, self.K, self.k, wob, rpm, q, self.rho, self.d_n, self.my, self.a11, self.a22, self.a33)
        if rop > self.last_rop:
            reward_1 = 1
        elif rop < self.last_rop:
            reward_1 = -1
        elif rop == 0:
            reward_1 = -1
        else:
            reward_1 = 0
        
        self.last_rop = rop
        self.depth += rop*self.delta_t
        reward = reward_1
        return reward, rop


    def render(self, mode = 'human'):
        self.plots.append(self.state)
        print(self.state)
        print(self.last_rop)
        print(self.reward)
        return self.plots
    
    def useless_counter(self, rop, counter):
        if rop == 0:
            counter += 1
        else:
            counter = 0
        return counter

    def reset(self):
        self.K = random.uniform(self.lb_k, self.ub_k)
        self.state = np.array([10,10,10,0], dtype = np.float32)
        self.depth_final = random.uniform(50, 250)
        self.depth = 0
        self.counter = 0
        print('reset')
        return self.state

class EckelEnv2(gym.Env):
    plots = []
    metadata = {'render.modes': ['human', 'ansi']}
    mode = 'human'
    MAX_WOB = 200
    MAX_RPM = 300
    MAX_Q = 400
    depth_final = random.uniform(50, 250)
    delta_t = 1 /3600
    v = 0.1  #feet/s
    a11 = 0.005
    a22 = 0.005
    a33 = 0.005
    a = 1
    b = 0.75
    c = 0.5
    k = 0.1
    rho = 1000
    d_n = 0.25
    my = 0.4
    ub_k = 0.9
    lb_k = 0.025
    state = [10,10,10,0]


    def __init__(self):
        self.viewer = True
        self.K = random.uniform(self.lb_k, self.ub_k)
        self.state = np.array([10,10,10,0], dtype = np.float32)
        self.reward = 0
        self.last_rop = 0
        self.counter = 0
        self.depth = 0
        high = np.array([self.MAX_WOB, self.MAX_RPM, self.MAX_Q, 100000], dtype = np.float32)
        low = np.array([0,0,0,0], dtype = np.float32)
        self.observation_space = spaces.Box(low = low, high = high, dtype = np.float32)
        #down, stay, up
        self.action_space = spaces.MultiDiscrete([3,3,3])


    
    def actionToValue(self, action):
        rates = np.zeros(3)
        for i in range(0, len(action)):
            if action[i] == 0:
                rates[i] = -1
            elif action[i] == 1:
                rates[i] = 0
            elif action[i] == 2:
                rates[i] = 1
        return rates
    
    def isDone(self):
        return not self.depth < self.depth_final

    def step(self, action):
        wob = self.state[0]
        rpm = self.state[1]
        q = self.state[2]
        rop = self.state[3]
        rates = self.actionToValue(action)
        reward, rop = self.alternative_reward(rates)
        self.state[0] += rates[0]
        self.state[1] += rates[1]
        self.state[2] += rates[2]
        self.state[3] = rop
        self.counter = self.useless_counter(rop,self.counter)
        done = self.isDone()
        if done:
            reward = + 1000
        if self.counter >= 100:
            reward = -1000
            done = True
        self.reward = reward
        return self.state, reward, done, {}




    def check_for_nan(self,number):
        return number != number

    def alternative_reward(self, rates):
        wob = self.state[0] + rates[0]
        rpm = self.state[1] + rates[1]
        q = self.state[2] + rates[2]
        depth = self.state[3]
        rop = rate_of_penetration_eckel_vary_founder(self.a, self.b, self.c, self.K, self.k, wob, rpm, q, self.rho, self.d_n, self.my, self.a11, self.a22, self.a33)
        if rop > self.last_rop:
            reward_1 = 1
        elif rop < self.last_rop:
            reward_1 = -1
        elif rop == 0:
            reward_1 = -1
        else:
            reward_1 = 0
        
        self.last_rop = rop
        self.depth += rop*self.delta_t
        reward = reward_1
        return reward, rop


    def render(self, mode = 'human'):
        self.plots.append(self.state)
        print(self.state)
        print(self.reward)
        return self.plots
    
    def useless_counter(self, rop, counter):
        if rop == 0:
            counter += 1
        else:
            counter = 0
        return counter

    def reset(self):
        self.K = random.uniform(self.lb_k, self.ub_k)
        #self.state = np.array([10,10,10,0], dtype = np.float32)
        self.state = self.state
        self.depth_final = random.uniform(50, 250)
        self.depth = 0
        self.counter = 0
        return self.state