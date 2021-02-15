from gym.envs.registration import register

register(id = 'test-v0',
        entry_point = 'testenv.envs:TestEnv')

        