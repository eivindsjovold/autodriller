import gym
import rop_envs

from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env('eckel-mdisc-v0', n_envs = 4)

model = A2C(MlpPolicy,env,verbose = 1)
model.learn(total_timesteps=100000)
model.save("a2c_mdisc")

obs = env.reset()
for i in range(0,50):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()