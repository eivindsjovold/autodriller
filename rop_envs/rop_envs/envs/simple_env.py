import numpy as np
import gym
import random
from gym import spaces

from models.simple_model import rop, rop_multi, rop_statements

class SimpleEnv1(gym.Env):
    plots = []
    metadata = {'render.modes': ['human', 'ansi']}
    mode = 'human'
    MAX_WOB = 200
    depth_final = 200


    def __init__(self):
        self.viewer = True
        self.wob_opt = 25
        self.state = np.array([0,0,0,0], dtype = np.float32)
        self.reward = 0
        self.delta_t = 1/3600
        high = np.array([self.MAX_WOB, self.MAX_WOB, 100000, 100000], dtype = np.float32)
        low = np.array([0,0,0,0], dtype = np.float32)
        self.observation_space = spaces.Box(low = low, high = high, dtype = np.float32)
        #down, stay, up
        self.action_space = spaces.Discrete(3)


    
    def actionToValue(self, action):
        if action == 0:
                rates = -1
        elif action == 1:
                rates = 0
        elif action == 2:
                rates = 1
        return rates
    
    def isDone(self):
        return not self.depth < self.depth_final

    def step(self, action):
        rates = self.actionToValue(action)
        #prev wob
        self.state[1] = self.state[0]
        #current wob
        self.state[0] += rates
        r = rop(self.state[0], self.wob_opt)
        self.state[3] = self.state[2]
        self.state[2] = r
        if self.state[2] > self.state[3]:
            reward_1 = 1
        elif self.state[2] < self.state[3]:
            reward_1 = -1
        else:
            reward_1 = 0

        self.depth += self.state[2]*self.delta_t
        reward = reward_1
        self.counter = self.useless_counter(rop,self.counter)
        done = self.isDone()

        if self.counter >= 10:
            reward = -1000
            done = True
        self.reward = reward
        return self.state, reward, done, self.wob_opt



    def check_for_nan(self,number):
        return number != number



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
        self.state = np.array([0,0,0,0], dtype = np.float32)
        self.depth = 0
        self.counter = 0
        return self.state



class SimpleEnv2(gym.Env):
    plots = []
    metadata = {'render.modes': ['human', 'ansi']}
    mode = 'human'
    MAX_WOB = 200
    depth_final = 200


    def __init__(self):
        self.viewer = True
        self.wob_opt = random.randint(10,1500)
        self.state = np.array([0,0,0,0], dtype = np.float32)
        self.reward = 0
        self.delta_t = 1/3600
        high = np.array([self.MAX_WOB, self.MAX_WOB, 100000, 100000], dtype = np.float32)
        low = np.array([0,0,0,0], dtype = np.float32)
        self.observation_space = spaces.Box(low = low, high = high, dtype = np.float32)
        #down, stay, up
        self.action_space = spaces.Discrete(3)
        self.best_value = 0


    
    def actionToValue(self, action):
        if action == 0:
                rates = -1
        elif action == 1:
                rates = 0
        elif action == 2:
                rates = 1
        return rates
    
    def isDone(self):
        return not self.depth < self.depth_final

    def step(self, action):
        rates = self.actionToValue(action)
        #prev wob
        self.state[1] = self.state[0]
        #current wob
        self.state[0] += rates
        r = rop(self.state[0], self.wob_opt)
        self.state[3] = self.state[2]
        self.state[2] = r
        if self.state[2] > self.state[3]:
            reward_1 = 1
        elif self.state[2] < self.state[3]:
            reward_1 = -1
        else:
            reward_1 = 0
        #if self.state[2] == self.wob_opt:
        #    reward_2 = 1
        #    #self.best_value = self.state[2]
        #else:
        #    reward_2 = 0
        self.depth += self.state[2]*self.delta_t
        reward = reward_1# + reward_2
        self.counter = self.useless_counter(rop,self.counter)
        done = self.isDone()

        if self.counter >= 10:
            reward = -1000
            done = True
        self.reward = reward
        return self.state, reward, done, self.wob_opt



    def check_for_nan(self,number):
        return number != number



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
        #self.state = np.array([0,0,0,0], dtype = np.float32)
        self.depth = 0
        self.counter = 0
        self.wob_opt = random.randint(10,1500)
        #print(self.wob_opt)
        return self.state

class SimpleEnv3(gym.Env):
    plots = []
    metadata = {'render.modes': ['human', 'ansi']}
    mode = 'human'
    MAX_WOB = 200/100
    MAX_Q = 400/100
    MAX_RPM = 300/100
    depth_final = 0.5


    def __init__(self):
        self.viewer = True
        self.K = 1.5
        self.wob_opt = self.K*100/100
        self.rpm_opt = self.K*150/100
        self.q_opt = self.K*200/100
        self.opt = [self.wob_opt, self.rpm_opt, self.q_opt]
        self.state = np.array([0,0,0,0,0,0,0,0], dtype = np.float32)
        self.reward = 0
        self.delta_t = 1/3600
        high = np.array([self.MAX_WOB, self.MAX_WOB, self.MAX_RPM, self.MAX_RPM,self.MAX_Q, self.MAX_Q,100000, 100000], dtype = np.float32)
        low = np.array([0,0,0,0,0,0,0,0], dtype = np.float32)
        self.observation_space = spaces.Box(low = low, high = high, dtype = np.float32)
        #down, stay, up
        self.action_space = spaces.MultiDiscrete([3,3,3])


    
    def actionToValue(self, action):
        rates = np.zeros(3)
        for i in range(0, len(action)):
            if action[i] == 0:
                rates[i] = -1/100
            elif action[i] == 1:
                rates[i] = 0
            elif action[i] == 2:
                rates[i] = 1/100
        return rates
    
    def isDone(self):
        return not self.depth < self.depth_final

    def step(self, action):
        rates = self.actionToValue(action)
        #prev wob
        self.state[1] = self.state[0]
        #current wob
        self.state[0] += rates[0]
        self.state[3] = self.state[2]
        self.state[2] += rates[1]
        self.state[5] = self.state[5]
        self.state[4] += rates[2] 
        
        r = self.K*rop_multi(self.state[0],self.state[2], self.state[4], self.wob_opt, self.rpm_opt, self.q_opt)
        self.state[7] = self.state[6]
        self.state[6] = r
        
        #if self.state[6] > self.state[7]:
        #    reward_1 = 1#self.state[6]/(self.wob_opt +self.q_opt+self.rpm_opt)
        #elif self.state[6] < self.state[7]:
        #    reward_1 = -1#self.state[6]/(self.wob_opt+self.q_opt+self.rpm_opt)
        #else:
        #    reward_1 = 0
        reward_1 = self.state[6] - self.state[7]
        self.depth += self.state[2]*self.delta_t
        reward = reward_1
        self.counter = self.useless_counter(self.state[6],self.counter)
        done = self.isDone()

        if self.counter >= 10:
            reward = -1000
            done = True

        self.reward = reward
        return self.state, reward, done, self.opt



    def check_for_nan(self,number):
        return number != number



    def render(self, mode = 'human'):
        self.plots.append(self.state)
        print(self.state)
        print(self.reward)
        return self.plots
    
    def useless_counter(self, dr, counter):
        if dr == 0:
            counter += 1
        else:
            counter = 0
        return counter

    def reset(self):
        #self.state = np.array([0,0,0,0,0,0,0,0], dtype = np.float32)
        self.depth = 0
        self.counter = 0
        self.K = random.uniform(0.1,1)
        print(self.K)
        return self.state
