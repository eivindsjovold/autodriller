from gym.envs.registration import register

register(id = 'by-v0',
        entry_point = 'rop_envs.envs:ByModEnv')
register(id = 'eckel-v0',
        entry_point = 'rop_envs.envs:EckelEnv')