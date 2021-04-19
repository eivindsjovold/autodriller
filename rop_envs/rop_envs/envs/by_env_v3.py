import numpy as np
import gym
import random
from gym import spaces

from models.bourgouyne_young_1974.rate_of_penetration_v3 import rate_of_penetration_modv3


class BYRate(gym.Env):
    MAX_WOB = 100
    MAX_RPM = 300
    MAX_Q = 400
    depth = 0.0 #feet
    rho = 22 
    wob_init = 0.1 #10^3 lbf/in 
    gp = 1.0        #lbm/gal
    db = 1.0
    db_init = db    #bit outer diameter in inches
    delta_t = 1/3600 #hrs
    h = 0.0001
    v = 0.1  #feet/s
    a11 = 0.05
    a22 = 0.05
    a33 = 0.00005
    tolerance = 4
 

    def __init__(self):
        self.depth_final = 5
        self.viewer = True
        self.state = np.array([5,5,0], dtype = np.float32)
        self.reward = 0
        self.rop_max = 0
        self.last_rop = 0
        self.plots = []
        self.num_it = 0
        high = np.array([self.MAX_WOB, self.MAX_RPM, 100000], dtype = np.float32)
        low = np.array([0,0,0], dtype = np.float32)
        self.observation_space = spaces.Box(low = low, high = high, dtype = np.float32)
        #down, stay, up
        self.action_space = spaces.MultiDiscrete([3,3])
        self.a1 = random.uniform(1.5,1.9)
        self.a2 = 0
        self.a3 = 0
        self.a4 = 0
        self.a5 = random.uniform(1,2)
        self.a6 = random.uniform(0.6,1)
        self.a7 = 0
        self.a8 = 0.3

    
    def actionToValue(self, action):
        rates = np.zeros(len(action))
        for i in range(0, len(action)):
            if action[i] == 0:
                rates[i] = -1
            elif action[i] == 1:
                rates[i] = 0
            elif action[i] == 2:
                rates[i] = 1
        return rates
    
    def check_for_nan(self,number):
        return number != number

    def check_wob(self, wob):
        if wob > self.MAX_WOB:
            return self.MAX_WOB
        else:
            return wob
    def check_rpm(self, rpm):
        if rpm > self.MAX_RPM:
            return self.MAX_RPM
        else:
            return rpm
    def check_q(self, q):
        if q > self.MAX_Q:
            return self.MAX_Q
        else:
            return q
    

    def step(self, action):
        depth = self.state[2]
        rates = self.actionToValue(action)
        wob = self.state[0] + rates[0]
        rpm = self.state[1] + rates[1]
        if wob > self.MAX_WOB:
            reward = -10
            wob = self.MAX_WOB
        if rpm > self.MAX_RPM:
            reward = -10
            rpm = self.MAX_RPM
        if wob <= 0:
            wob = 0
            reward = -10
        if rpm <= 0:
            rpm = 0
            reward = -10
        self.state[0] = wob
        self.state[1] = rpm
        rop = rate_of_penetration_modv3(self.a1,self.a2,self.a3,self.a4,self.a5,self.a6,self.a7,self.a8,depth,self.gp,self.rho,wob,self.wob_init,self.db,self.db_init,rpm,self.h,400,self.v, self.a11, self.a22, self.a33)
        if self.check_for_nan(rop):
            rop = 0
        self.state[2] += rop*self.delta_t            
        
        if rop > self.last_rop:
            reward_1 = 1
        elif rop == self.last_rop:
            reward_1 = 0
        else:
            reward_1 = -1

        self.last_rop = rop
        self.num_it += 1
        done = self.isDone()
        if done:
            reward_2 = 10000 - self.num_it
        else:
            reward_2 = 0
        reward = reward_1 + reward_2 
        self.reward = reward
        return self.state, reward, done, self.last_rop

    def isDone(self):
        return not self.state[2] < self.depth_final

    def render(self, mode = 'human'):
        self.plots.append(self.state)
        print(self.state)
        print(self.last_rop)
        return self.plots

    def reset(self,a1,a5):
        self.state = np.array([5,5,0], dtype = np.float32)
        self.rop_max = 0
        self.reward = 0
        self.a1 = random.uniform(1.5,1.9)
        self.a1 = a1
        self.a5 = random.uniform(1.5,2)
        self.a5 = a5
        self.a2 = 0
        self.a3 = 0
        self.a4 = 0
        self.a6 = 1
        self.a8 = 0.3
        self.num_it = 0
        self.depth_final = random.randint(10,50)
        return self.state