from stable_baselines3.common.env_checker import check_env
import gym
import testenv

env = gym.make('test-v0')
check_env(env)