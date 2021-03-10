import numpy as np
import gym
from gym import spaces

from models.bourgouyne_young_1974.rate_of_penetration_modified import rate_of_penetration_mod
from models.bourgouyne_young_1974.read_from_case import read_from_case
from models.eckels.eckel import rate_of_penetration_eckel

class EckelRate(gym.Env):
    plots = []
    metadata = {'render.modes': ['human', 'ansi']}
    mode = 'human'
    MAX_WOB = 300
    MAX_RPM = 300
    MAX_Q = 400
    depth_final = 10
    delta_t = 1 /3600
    v = 0.1  #feet/s
    a = 1
    b = 1
    c = 1
    K = 1
    k = 1
    my = 0.4
    d_n = 1.0
    rho = 22
    a11 = 0.005
    a22 = 0.005
    a33 = 0.00005




    def __init__(self):
        self.viewer = True
        self.state = np.array([15,15,100,0], dtype = np.float32)
        self.reward = 0
        self.last_rop = 0
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
        return not self.state[3] < self.depth_final

    def step(self, action):
        wob = self.state[0]
        rpm = self.state[1]
        q = self.state[2]
        depth = self.state[3]
        rates = self.actionToValue(action)
        reward, depth = self.alternative_reward(rates)
        self.state[0] += rates[0]
        self.state[1] += rates[1]
        self.state[2] += rates[2]
        self.state[3] = depth
        done = self.isDone()
        self.reward = reward
        return self.state, reward, done, {}



    def calculateReward(self, rates):
        wob = self.state[0] + rates[0]
        rpm = self.state[1] + rates[1]
        q = self.state[2] + rates[2]
        depth = self.state[3]
        if wob <= 0:
            wob = 0
            reward = -10
        elif rpm <= 0:
            rpm = 0
            reward = -10
        elif q <= 0:
            q = 0
            reward = -10
        else:    
            rop = rate_of_penetration_eckel(self.a, self.b, self.c, self.K, self.k, wob, rpm, q, self.rho, self.d_n, self.my, self.a11, self.a22, self.a33)
            depth = self.state[3] + rop*self.delta_t
            reward = rop
        return reward, depth

    def alternative_reward(self, rates):
        wob = self.state[0] + rates[0]
        rpm = self.state[1] + rates[1]
        q = self.state[2] + rates[2]
        depth = self.state[3]
        rop = rate_of_penetration_eckel(self.a, self.b, self.c, self.K, self.k, wob, rpm, q, self.rho, self.d_n, self.my, self.a11, self.a22, self.a33)
        if rop < self.last_rop:
            reward = -1
        elif rop == self.last_rop:
            reward = 0
        elif rop > self.last_rop: 
            reward = 1
        else:
            print('entered an unexpected state of reward, reward set to zero')
            reward = 0
        self.last_rop = rop
        depth += rop*self.delta_t
        return reward, depth


    def render(self, mode = 'human'):
        self.plots.append(self.state)
        print(self.state)
        print(self.last_rop)
        return self.plots

    def reset(self):
        print('Resetting: ')
        self.state = np.array([15,15,100,0], dtype = np.float32)
        return self.state


class BYRate(gym.Env):
    MAX_WOB = 100
    MAX_RPM = 100
    MAX_Q = 334
    depth = 0.0 #feet
    depth_final = 1000.0 #feet
    rho = 22 
    q = 0     #gal/min
    wob_init = 0.1 #10^3 lbf/in
    wob = 50      #10^3 lbf/in 
    gp = 1.0        #lbm/gal
    db = 1.0
    db_init = db    #bit outer diameter in inches
    delta_t = 1/3600 #hrs
    h = 0.0001
    a1 = 1.831
    a2 = 0.000075
    a3 = 0.000005
    a4 = 0.000005
    a5 = 0.865
    a6 = 1
    a7 = 0.305
    a8 = 0.5
    v = 0.1  #feet/s
    a11 = 0.005
    a22 = 0.005
    a33 = 0.00005

    def __init__(self):
        self.viewer = True
        self.state = np.array([15,15,100,0], dtype = np.float32)
        self.reward = 0
        self.last_rop = 0
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
    def calculate_reward(self, rates):
        wob = self.state[0] + rates[0]
        rpm = self.state[1] + rates[1]
        q = self.state[2] + rates[2]
        depth = self.state[3]
        rop = rate_of_penetration_mod(self.a1,self.a2,self.a3,self.a4,self.a5,self.a6,self.a7,self.a8,depth,self.gp,self.rho,wob,self.wob_init,self.db,self.db_init,rpm,self.h,q,self.v, self.a11, self.a22, self.a33)
        if rop < self.last_rop:
            reward = -1
        elif rop == self.last_rop:
            reward = 0
        elif rop > self.last_rop: 
            reward = 1
        else:
            print('entered an unexpected state of reward, reward set to zero')
            reward = 0
        self.last_rop = rop
        depth += rop*self.delta_t
        return reward, depth



    def step(self, action):
        wob = self.state[0]
        rpm = self.state[1]
        q = self.state[2]
        depth = self.state[3]
        rates = self.actionToValue(action)
        reward, depth = self.calculate_reward(rates)
        self.state[0] += rates[0]
        self.state[1] += rates[1]
        self.state[2] += rates[2]
        self.state[3] = depth
        done = self.isDone()
        self.reward = reward
        return self.state, reward, done, {}

    def isDone(self):
        return not self.state[3] < self.depth_final

    def render(self, mode = 'human'):
        self.plots.append(self.state)
        print(self.state)
        print(self.last_rop)
        return self.plots

    def reset(self):
        print('Resetting: ')
        self.state = np.array([15,15,100,0], dtype = np.float32)
        return self.state
