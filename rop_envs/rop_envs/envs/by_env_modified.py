import numpy as np
import gym
from gym import spaces

from models.bourgouyne_young_1974.rate_of_penetration_modified import rate_of_penetration_mod
from models.bourgouyne_young_1974.read_from_case import read_from_case

class ByModEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}
    mode = 'human'    

    MAX_WOB = 20
    MAX_RPM = 650
    MAX_Q = 400

    MIN_WOB = 5
    MIN_RPM = 150
    MIN_Q = 100

    def __init__(self):
        self.viewer = None
        self.state = np.zeros(2)
        self.seed()

        high = np.array([self.MAX_WOB, self.MAX_RPM, self.MAX_Q], dtype=np.float32)
        low = np.array([self.MIN_WOB, self.MIN_RPM, self.MIN_Q], dtype=np.float32)
        self.observation_space = spaces.Box(low = 0, high = np.inf, shape = (2,), dtype=np.float32)
        self.action_space = spaces.Box(low = low, high = high, dtype = np.float32)        


        ## Drilling parameters
        self.depth_final = 500 #feet
        self.rho = 22 
        self.wob_init = 0.1 #10^3 lbf/in
        self.gp = 1.0        #lbm/gal
        self.db = 1.0
        self.db_init = self.db    #bit outer diameter in inches
        self.delta_t = 1/3600 #hrs
        self.h = 0.0001
        self.velocity_noz = 0.1
        self.a11 = 0.05
        self.a22 = 0.05
        self.a33 = 0.05

        # Formation parameters
        formation_change = read_from_case()
        self.a1 = formation_change[1][6]
        self.a2 = formation_change[2][6]
        self.a3 = formation_change[3][6]
        self.a4 = 0#formation_change[4][6]
        self.a5 = formation_change[5][6]
        self.a6 = formation_change[6][6]
        self.a7 = 0#formation_change[7][6]
        self.a8 = formation_change[8][6]


    def step(self, action):
        depth = self.state[0]
        rpm = action[1]
        wob = action[0]
        q = action[2]
        rop = rate_of_penetration_mod(self.a1,self.a2,self.a3,self.a4,self.a5,self.a6,self.a7,self.a8,depth,self.gp,self.rho,wob,self.wob_init,self.db,self.db_init,rpm,self.h,q,self.velocity_noz, self.a11, self.a22, self.a33)
        depth += rop*self.delta_t
        self.state = np.array([depth, rop])
        done = self.isDone()
        reward = rop/1000

        return self.state, reward, done, {}
    
    def render(self, mode = 'human'):
        print( str(' Depth : ') + str(self.state[0]) + str('\n'))
        print( str(' ROP : ') + str(self.state[1]) + str('\n'))

    
    def reset(self):
        self.state = np.zeros(2)
        return self.state

    def isDone(self):
        if self.depth_final <= self.state[0]:
            return False
        else:
            return True