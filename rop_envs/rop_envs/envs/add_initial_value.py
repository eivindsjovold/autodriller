import numpy as np
import gym
import random
from gym import spaces

from models.bourgouyne_young_1974.rate_of_penetration_modified import rate_of_penetration_mod
from models.bourgouyne_young_1974.rate_of_penetration import rate_of_penetration
from models.bourgouyne_young_1974.read_from_case import read_from_case
from models.eckels.eckel import rate_of_penetration_eckel

class EckelRateIV(gym.Env):
    plots = []
    metadata = {'render.modes': ['human', 'ansi']}
    mode = 'human'
    MAX_WOB = 300
    MAX_RPM = 300
    MAX_Q = 400
    depth_final = random.uniform(10, 250)
    delta_t = 1 /3600
    v = 0.1  #feet/s
    a11 = 0.005
    a22 = 0.005
    a33 = 0.00005
    a = 0.1
    b = 0.11
    c = 0.5
    k = 0.1
    rho = 1000
    d_n = 0.25
    my = 0.4
    ub_k = 0.3
    lb_k = 0.1


    def __init__(self):
        self.viewer = True
        #self.disturb = 10
        #self.formation_depth = 0
        self.K = random.uniform(self.lb_k, self.ub_k)
        self.state = np.array([0,0,0,0], dtype = np.float32)
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
        #if self.formation_depth > self.disturb:
        #    self.change_formation()
        return self.state, reward, done, {}

    #def change_formation(self):
    #    self.K = random.uniform(self.lb_k, self.ub_k)
    #    self.formation_depth = 0
    #    self.disturb = random.uniform(self.state[3] + 5, self.state[3] + 100 )


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
            self.formation_depth += rop*self.delta_t
            reward = rop
        return reward, depth

    def check_for_nan(self,number):
        return number != number

    def alternative_reward(self, rates):
        wob = self.state[0] + rates[0]
        rpm = self.state[1] + rates[1]
        q = self.state[2] + rates[2]
        depth = self.state[3]
        rop = rate_of_penetration_eckel(self.a, self.b, self.c, self.K, self.k, wob, rpm, q, self.rho, self.d_n, self.my, self.a11, self.a22, self.a33)
        if self.check_for_nan(rop):
            reward_1 = -10
            rop = 0
            depth = self.depth_final + 10
        reward_1 = rop - self.last_rop
        
        self.last_rop = rop
        depth += rop*self.delta_t
        reward = reward_1# + reward_2
        return reward, depth


    def render(self, mode = 'human'):
        self.plots.append(self.state)
        print(self.state)
        print(self.last_rop)
        print(self.reward)
        return self.plots
    
    def detect_change(self, rop):
        if rop > self.last_rop + 10 or rop < self.last_rop -10:
            return True
        else:
            return False


    def reset(self):
        self.K = random.uniform(self.lb_k, self.ub_k)
        self.state = self.state
        return self.state
