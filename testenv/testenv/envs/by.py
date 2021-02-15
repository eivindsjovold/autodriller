import numpy as np
import gym
from gym import spaces
from models.bourgouyne_young_1974.rate_of_penetration import rate_of_penetration
from models.bourgouyne_young_1974.read_from_case import read_from_case

class TestEnv(gym.Env):

    MAX_WOB = 300
    MAX_RPM = 300
    MAX_Q = 300
    MAX_ROP = 300

    MIN_WOB = 150
    MIN_RPM = 150
    MIN_Q = 150

    def __init__(self):
        self.viewer = None
        self.state = np.zeros(2)
        self.seed()
        
        high = np.array([self.MAX_WOB, self.MAX_RPM, self.MAX_Q], dtype=np.float32)
        low = np.array([self.MIN_WOB, self.MIN_RPM, self.MIN_Q], dtype=np.float32)
        self.observation_space = spaces.Box(low = 0, high = np.inf, shape = (2,), dtype=np.float32)
        self.action_space = spaces.Box(low = low, high = high, dtype = np.float32)
        
        ## Formation parameters
        formation_change = read_from_case()
        self.a1 = formation_change[1][1]
        self.a2 = formation_change[1][2]
        self.a3 = formation_change[1][3]
        self.a4 = formation_change[1][4]
        self.a5 = formation_change[1][5]
        self.a6 = formation_change[1][6]
        self.a7 = formation_change[1][7]
        self.a8 = formation_change[1][8]

        
        ## Drilling parameters
        self.depth_final = 500.0 #feet
        self.rho = 22 
        self.wob_init = 0.1 #10^3 lbf/in
        self.gp = 1.0        #lbm/gal
        self.db = 1.0
        self.db_init = self.db    #bit outer diameter in inches
        self.delta_t = 1/3600 #hrs
        self.h = 0.0001
        self.velocity_noz = 1
    
    def isDone(self):
        if self.state[0] >= self.depth_final:
            return True
        else: 
            return False

    def step(self, act_input):
        depth = self.state[0]

        rpm =  act_input[0]
        wob = act_input[1]
        q = act_input[2] 
        rop = rate_of_penetration(self.a1,self.a2,self.a3,self.a4,self.a5,self.a6,self.a7,self.a8,depth,self.gp,self.rho,wob,self.wob_init,self.db,self.db_init,rpm,self.h,q,self.velocity_noz)
        depth += rop*self.delta_t
        self.state = np.array([depth, rop])
        reward = rop/1000
        done = self.isDone()
        return self.state, reward, done, {}
        
    def reset(self):
        self.state = np.zeros(2)
        return self.state

    def render(self):
        print( str(' Depth : ') + self.state[0] + str('\n'))
        print( str(' ROP : ') + self.state[1] + str('\n'))

        
