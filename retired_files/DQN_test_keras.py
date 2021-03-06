import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
eckel = 'eckel-v0'

env = gym.make(eckel)

np.random.seed(123)
env.seed(123)
nb_actions = env.action_space

model = Sequential()
model.add(Flatten(input_shape = (3,) + env.observation_space))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

memory = SequentialMemory(limit = 50000, window_length = 1)
policy = BoltzmannQPolicy()

dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=10, visualize=False, verbose=2)

dqn.save_weights('dqn_{}_weights.h5f'.format(eckel), overwrite=True)

dqn.test(env, nb_episodes=1, visualize=True)