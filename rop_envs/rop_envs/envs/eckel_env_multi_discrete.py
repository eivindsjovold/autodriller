import numpy as np
import gym
from gym import spaces
from models.eckels.eckel import rate_of_penetration_eckel


class EckelEnvMDisc(gym.Env):

    metadata = {'render.modes': ['human', 'ansi']}
    
    MAX_WOB = 170
    MAX_RPM = 167.5
    MAX_Q = 324

    MIN_WOB = 0
    MIN_RPM = 40
    MIN_Q = 100
    

    def __init__(self):
        self.viewer = None
        self.state = np.zeros(2)
        self.seed()
        
        #Eckel
        self.depth_final = 5
        self.delta_t = 1 /3600
        self.v = 0.1  #feet/s
        self.a = 1
        self.b = 1
        self.c = 1
        self.K = 1
        self.k = 1
        self.my = 0.4
        self.d_n = 1.0
        self.rho = 22
        self.a11 = 0.005
        self.a22 = 0.005
        self.a33 = 0.05

        #high = np.array([self.MAX_WOB, self.MAX_RPM, self.MAX_Q], dtype=np.float32)
        #low = np.array([self.MIN_WOB, self.MIN_RPM, self.MIN_Q], dtype=np.float32)
        self.observation_space = spaces.Box(low = 0, high = np.inf, shape = (2,), dtype=np.float32)
        ###NB Blir negativ på 171 med dette opsettet.
        self.action_space = spaces.MultiDiscrete([self.MAX_WOB, self.MAX_RPM, self.MAX_Q])
        ###NB Max WOB i denne casen: 117.61340377001153


    def step(self,action):
        wob = action[0]
        rpm = action[1]
        q = action[2] + 10 #for å unngå negativ verdi
        depth = self.state[0]
        rop = rate_of_penetration_eckel(self.a,self.b,self.c,self.K, self.k, wob, rpm, q, self.rho, self.d_n, self.my, self.a11, self.a11, self.a22)
        depth += rop*self.delta_t
        self.state[0] = depth
        self.state[1] = rop
        done = self.isDone()
        reward = rop/100
        return self.state, reward, done, {}

    def render(self, mode = 'human'):
        print( str(' Depth : ') + str(self.state[0]) + str('\n'))
        print( str(' ROP : ') + str(self.state[1]) + str('\n'))

    def reset(self):
        self.state = np.zeros(2)
        return self.state

    def isDone(self):
        if self.depth_final >= self.state[0]:
            return False
        else:
            return True