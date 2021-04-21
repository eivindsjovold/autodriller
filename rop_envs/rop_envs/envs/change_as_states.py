import numpy as np
import gym
import random
from gym import spaces

from models.bourgouyne_young_1974.rate_of_penetration_modified import rate_of_penetration_mod
from models.bourgouyne_young_1974.rate_of_penetration import rate_of_penetration
from models.bourgouyne_young_1974.read_from_case import read_from_case
from models.eckels.eckel import rate_of_penetration_eckel
from models.eckels.eckel import rate_of_penetration_eckel_individual_founder


class EckelRate(gym.Env):
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
        return not self.state[3] < self.depth_final

    def step(self, action):
        wob = self.state[0]
        rpm = self.state[1]
        q = self.state[2]
        depth = self.state[3]
        rates = self.actionToValue(action)
        reward, depth, rop = self.alternative_reward(rates)
        self.state[0] += rates[0]
        self.state[1] += rates[1]
        self.state[2] += rates[2]
        self.state[3] = depth
        self.counter = self.useless_counter(rop,self.counter)
        done = self.isDone()
        if done:
            reward = + 1000
        if self.counter >= 10:
            reward = -1000
            done = True
        self.reward = reward
        return self.state, reward, done, rop

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
        depth += rop*self.delta_t
        reward = reward_1
        '''
        if q > self.MAX_Q:
            q = self.MAX_Q 
            reward = -10
        elif q < 0:
            q = 0
            reward = -10
        if wob > self.MAX_WOB:
            wob = self.MAX_WOB
            reward = -10
        elif wob < 0:
            wob = 0
            reward = -10
        if rpm > self.MAX_RPM:
            rpm = self.MAX_RPM
            reward = -10
        elif rpm < 0:
            rpm = 0
            reward = -10
        '''
        return reward, depth, rop


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
        self.counter = 0
        print('reset')
        return self.state

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
        rop = rate_of_penetration_mod(self.a1,self.a2,self.a3,self.a4,self.a5,self.a6,self.a7,self.a8,depth,self.gp,self.rho,wob,self.wob_init,self.db,self.db_init,rpm,self.h,400,self.v, self.a11, self.a22, self.a33)
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
        return self.state, reward, done, {}

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


class BenchmarkEckel(gym.Env):
    plots = []
    metadata = {'render.modes': ['human', 'ansi']}
    mode = 'human'
    MAX_WOB = 300
    MAX_RPM = 300
    MAX_Q = 400
    depth_final = 100
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
    k_1 = 0.15
    k_2 = 0.234




    def __init__(self):
        self.viewer = True
        #self.disturb = 10
        #self.formation_depth = 0
        self.K = 0
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
        return self.state, reward, done, [self.last_rop, self.K]

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

    def alternative_reward(self, rates):
        wob = self.state[0] + rates[0]
        rpm = self.state[1] + rates[1]
        q = self.state[2] + rates[2]
        depth = self.state[3]
        rop = rate_of_penetration_eckel(self.a, self.b, self.c, self.K, self.k, wob, rpm, q, self.rho, self.d_n, self.my, self.a11, self.a22, self.a33)
        if rop < self.last_rop:
            reward_1 = -1
        elif rop == self.last_rop:
            reward_1 = 0
        elif rop > self.last_rop: 
            reward_1 = 1
        else:
            print('entered an unexpected state of reward, reward set to zero')
            print(rop)
            print(depth)
            reward_1 = 0
        #if rop < 0:
        #    reward_2 = -100
        #else:
        #    reward_2 = 0    

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

    def reset(self, K):
        print('Resetting: ')
        self.K = K
        print('Hardness: ', self.K)
        self.state = np.array([15,15,100,0], dtype = np.float32)
        return self.state