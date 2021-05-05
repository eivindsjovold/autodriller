import numpy as np
import gym
import random
from gym import spaces

from models.bourgouyne_young_1974.rate_of_penetration_v3 import rate_of_penetration_modv3, f5, f6,f8


class BYEnv(gym.Env):
    MAX_WOB = 200
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
    v = 10  #feet/s
    a11 = 0.005
    a22 = 0.005
    a33 = 0.005
    tolerance = 4
 

    def __init__(self):
        self.depth_final = 50
        self.depth = 0
        self.viewer = True
        self.state = np.array([50,50,50,0], dtype = np.float32)
        self.reward = 0
        self.rop_max = 0
        self.last_rop = 0
        self.num_it = 0
        high = np.array([self.MAX_WOB, self.MAX_RPM, self.MAX_Q, 10000], dtype = np.float32)
        low = np.array([0,0,0,0], dtype = np.float32)
        self.observation_space = spaces.Box(low = low, high = high, dtype = np.float32)
        self.action_space = spaces.MultiDiscrete([3,3,3])
        self.a1 = random.uniform(1.0,1.5)
        self.a2 = 0
        self.a3 = 0
        self.a4 = 0
        self.a5 = (2-self.a1)
        self.a6 = (self.a1*0.5)
        self.a7 = 0
        self.a8 = self.a1-0.9
        self.rop_threshold = 0
        self.counter = 0

    
    def actionToValue(self, action):
        rates = np.zeros(len(action))
        for i in range(0,len(action)):
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
        depth = self.depth
        rates = self.actionToValue(action)
        wob = self.state[0] + rates[0]
        rpm = self.state[1] + rates[1]
        q = self.state[2]   + rates[2]
        rop = rate_of_penetration_modv3(self.a1,self.a2,self.a3,self.a4,self.a5,self.a6,self.a7,self.a8,depth,self.gp,self.rho,wob,self.wob_init,self.db,self.db_init,rpm,self.h,q,self.v, self.a11, self.a22, self.a33)
        if self.check_for_nan(rop):
            rop = 0
        self.depth += rop*self.delta_t            
        self.state[0] = wob
        self.state[1] = rpm
        self.state[2] = q
        self.state[3] = rop     
        
        
        if rop > self.last_rop:
            reward_1 = 1
        elif rop == self.last_rop:
            reward_1 = 0
        else:
            reward_1 = -1
        
        if rop > self.rop_threshold:
            reward_1 += 1
            self.rop_threshold = rop

        self.last_rop = rop
        self.num_it += 1
        done = self.isDone()
        
        if done:
            reward_2 = 100
        else:
            reward_2 = 0
        reward = reward_1# + reward_2
        
        self.counter = self.useless_counter(rop,self.counter)
        

        if self.counter >= 100:
            reward = -1000
            done = True
            self.state = np.array([50,50,50,0],dtype = np.float32)
 
        self.reward = reward
        return self.state, reward, done, {}

    def isDone(self):
        return not self.depth < self.depth_final

    def render(self, mode = 'human'):
        print(self.state)
    
    def useless_counter(self, rop, counter):
        if rop == 0:
            counter += 1
        else:
            counter = 0
        return counter

    def reset(self):
        random_reset = random.uniform(1,20)
        #if random_reset == 10:
        #    print('random')
        #    self.state = np.array([10,10,10,0],dtype=np.float32)
        #else:
        self.state = np.array([self.state[0],self.state[1],self.state[2],self.state[3]], dtype = np.float32)
        self.rop_max = 0
        self.reward = 0
        self.a1 = 1
        self.a5 =(2-self.a1)
        self.a2 = 0
        self.a3 = 0
        self.a4 = 0
        self.a6 = (self.a1*0.5)
        self.a8 = self.a1-0.9
        self.num_it = 0
        self.depth = 0
        self.depth_final = random.randint(50,250)
        self.depth_final = 100
        self.counter = 0
        self.rop_threshold = 0
        return self.state



class BYTestEnv(gym.Env):
    MAX_WOB = 200
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
    v = 10  #feet/s
    a11 = 0.005
    a22 = 0.005
    a33 = 0.005
    tolerance = 4
 

    def __init__(self):
        self.depth_final = 50
        self.depth = 0
        self.viewer = True
        self.state = np.array([10,10,10,0], dtype = np.float32)
        self.reward = 0
        self.rop_max = 0
        self.last_rop = 0
        self.num_it = 0
        high = np.array([self.MAX_WOB, self.MAX_RPM, self.MAX_Q, 10000], dtype = np.float32)
        low = np.array([0,0,0,0], dtype = np.float32)
        self.observation_space = spaces.Box(low = low, high = high, dtype = np.float32)
        self.action_space = spaces.MultiDiscrete([3,3,3])
        self.a1 = random.uniform(1.0,1.5)
        self.a2 = 0
        self.a3 = 0
        self.a4 = 0
        self.a5 = (2-self.a1)
        self.a6 = (self.a1*0.5)
        self.a7 = 0
        self.a8 = self.a1-0.9
        self.counter = 0

    
    def actionToValue(self, action):
        rates = np.zeros(len(action))
        for i in range(0,len(action)):
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
        depth = self.depth
        rates = self.actionToValue(action)
        wob = self.state[0] + rates[0]
        rpm = self.state[1] + rates[1]
        q = self.state[2]   + rates[2]
        rop = rate_of_penetration_modv3(self.a1,self.a2,self.a3,self.a4,self.a5,self.a6,self.a7,self.a8,depth,self.gp,self.rho,wob,self.wob_init,self.db,self.db_init,rpm,self.h,q,self.v, self.a11, self.a22, self.a33)
        if self.check_for_nan(rop):
            rop = 0

        self.depth += rop*self.delta_t            
        self.state[0] = wob
        self.state[1] = rpm
        self.state[2] = q
        self.state[3] = rop        
        
        if rop > self.last_rop:
            reward_1 = 1
        elif rop == self.last_rop:
            reward_1 = 0
        else:
            reward_1 = -1

        self.last_rop = rop
        self.num_it += 1
        done = self.isDone()
        
        #if done:
        #    reward_2 = 100
        #else:
        #    reward_2 = 0
        reward = reward_1# + reward_2
        
        self.counter = self.useless_counter(rop,self.counter)
        
        if self.counter >= 100:
            reward = -100
            done = True
            self.state = np.array([10,10,10,0],dtype = np.float32)
 
        self.reward = reward
        return self.state, reward, done, {}

    def isDone(self):
        return not self.depth < self.depth_final

    def render(self, mode = 'human'):
        print(self.state)
    
    def useless_counter(self, rop, counter):
        if rop == 0:
            counter += 1
        else:
            counter = 0
        return counter

    def reset(self, a1,length, init_vals):
        self.state = init_vals
        self.rop_max = 0
        self.reward = 0
        self.a1 = a1
        self.a5 = (2-a1)
        self.a2 = 0
        self.a3 = 0
        self.a4 = 0
        self.a6 = (a1*0.5)
        self.a8 = (a1-0.9)
        self.num_it = 0
        self.depth = 0
        self.depth_final = length
        self.counter = 0
        return self.state

class BYMemory(gym.Env):
    MAX_WOB = 200
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
    v = 10  #feet/s
    a11 = 0.005
    a22 = 0.005
    a33 = 0.005
    tolerance = 4
 

    def __init__(self):
        self.depth_final = 90
        self.depth = 0
        self.viewer = True
        self.state = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype = np.float32)
        self.reward = 0
        self.rop_max = 0
        self.last_rop = 0
        self.num_it = 0
        high = np.array([self.MAX_WOB, self.MAX_WOB, self.MAX_WOB, self.MAX_WOB,\
        self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM,\
        self.MAX_Q, self.MAX_Q, self.MAX_Q, self.MAX_Q,\
        10000, 10000,10000,10000], dtype = np.float32)

        low = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype = np.float32)
        self.observation_space = spaces.Box(low = low, high = high, dtype = np.float32)
        self.action_space = spaces.MultiDiscrete([3,3,3])
        self.a1 = 1
        self.a2 = 0
        self.a3 = 0
        self.a4 = 0
        self.a5 = 1#(2-self.a1)
        self.a6 = 0.75#(self.a1*0.5)
        self.a7 = 0
        self.a8 = 0.1#self.a1-0.9
        self.rop_threshold = 0
        self.counter = 0

    
    def actionToValue(self, action):
        rates = np.zeros(len(action))
        for i in range(0,len(action)):
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
        if wob > self.MAX_WOB or wob < 0:
            reward_2 = -10
        else:
            reward_2 = 0
        return reward_2
    def check_rpm(self, rpm):
        if rpm > self.MAX_RPM or rpm < 0:
            reward_2 = -10
        else:
            reward_2 = 0
        return reward_2
    def check_q(self, q):
        if q > self.MAX_Q or q < 0:
            reward_2 = -10
        else:
            reward_2 = 0
        return reward_2

    def step(self, action):
        
        for i in range(0,-3,-1):
            #wob memory
            self.state[i+3] = self.state[i+2]
            #rpm memory
            self.state[i+7] = self.state[i+6]
            #q memory
            self.state[i+11] = self.state[i+10]
            #rop memory
            self.state[i+15] = self.state[i+14]

        depth = self.depth
        rates = self.actionToValue(action)
        self.state[0] += rates[0]
        self.state[4] += rates[1]
        self.state[8] += rates[2]
        rop = rate_of_penetration_modv3(self.a1,self.a2,self.a3,self.a4,self.a5,self.a6,self.a7,self.a8,depth,self.gp,self.rho,self.state[0],self.wob_init,self.db,self.db_init,self.state[4] ,self.h,self.state[8],self.v, self.a11, self.a22, self.a33)
        
        #some hard constraints on rop and depth
        if self.check_for_nan(rop):
            rop = 0
        if rop < 0:
            self.depth += 0
        else:    
            self.depth += rop*self.delta_t            
        
        #soft constraints on input variables

        self.state[12] = rop
        reward_2 = 0     
        
        difference = np.diff([self.state[13], self.state[12]])
        reward_1 = difference[0]
        

        self.last_rop = rop
        self.num_it += 1
        done = self.isDone()
        
        #if done == True:
        #    print(self.state)
        #    reward_2 = 100000*(1/self.num_it)
        #else:
        #    reward_2 = 0
        reward = reward_1# + reward_2
        
        self.counter = self.useless_counter(rop,self.counter)
        #if done == True:
        #    print(self.state)

        if self.counter >= 10:
            reward = -1000
            done = True
            #self.state = np.array([50,50,50,0],dtype = np.float32)
 
        self.reward = reward
        return self.state, reward, done, {}

    def isDone(self):
        return not self.depth < self.depth_final

    def render(self, mode = 'human'):
        print(self.state)
    
    def useless_counter(self, rop, counter):
        if rop <= 0:
            counter += 1
        else:
            counter = 0
        return counter

    def reset(self):
        self.state = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype = np.float32)
        self.rop_max = 0
        self.reward = 0
        self.num_it = 0
        self.depth = 0
        self.counter = 0
        self.rop_threshold = 0
        return self.state


class WOBEnv1(gym.Env):
    MAX_WOB = 200
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
    v = 10  #feet/s
    a11 = 0.005
    a22 = 0.005
    a33 = 0.005
    tolerance = 4

    def __init__(self):
        self.depth_final = 150
        self.depth = 0
        self.viewer = True
        self.state = np.array([10,10,10,10], dtype = np.float32)
        self.reward = 0
        self.rop_max = 0
        self.last_rop = 0
        self.num_it = 0
        high = np.array([self.MAX_WOB, self.MAX_WOB,10000, 10000], dtype = np.float32)
        low = np.array([0,0,0,0], dtype = np.float32)
        self.observation_space = spaces.Box(low = low, high = high, dtype = np.float32)
        self.action_space = spaces.Discrete(3)
        #self.a1 = random.uniform(1.0,1.5)
        self.a1 = 1
        self.a2 = 0
        self.a3 = 0
        self.a4 = 0
        self.a5 = 1#(2-self.a1)
        self.a6 = 0.75#(self.a1*0.5)
        self.a7 = 0
        self.a8 = 0.4#self.a1-0.9
        self.rop_threshold = 0
        self.counter = 0

    
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
        r = f5(self.a5, self.state[0], self.wob_init, self.db, self.db_init,self.a11)
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
        self.counter = self.useless_counter(r,self.counter)
        done = self.isDone()

        if self.counter >= 10:
            reward = -1000
            done = True
        self.reward = reward
        print(self.state)
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
        self.state = np.array([10,10,0,0], dtype = np.float32)
        self.depth = 0
        self.counter = 0
        return self.state



class RPMEnv1(gym.Env):
    plots = []
    metadata = {'render.modes': ['human', 'ansi']}
    mode = 'human'
    MAX_WOB = 200
    depth_final = 100


    def __init__(self):
        self.viewer = True
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

class QEnv1(gym.Env):
    plots = []
    metadata = {'render.modes': ['human', 'ansi']}
    mode = 'human'
    MAX_WOB = 200
    depth_final = 100


    def __init__(self):
        self.viewer = True
        self.wob_opt = 300
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