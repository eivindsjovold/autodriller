import numpy as np
import gym
import random
from gym import spaces

from models.simple_model import rop

class SimpleEnv1(gym.Env):
    plots = []
    metadata = {'render.modes': ['human', 'ansi']}
    mode = 'human'
    MAX_WOB = 200
    depth_final = 200


    def __init__(self):
        self.viewer = True
        self.wob_opt = 100
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
        return self.state, reward, done, {}



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
        self.wob_opt = random.randint(100,150)
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
        self.wob_opt = random.randint(100,150)
        #print(self.wob_opt)
        return self.state