from gym.envs.registration import register

register(id = 'by-v0',
        entry_point = 'rop_envs.envs:ByModEnv')
register(id = 'eckel-v0',
        entry_point = 'rop_envs.envs:EckelEnv')
register(id = 'eckel-disc-v0',
        entry_point = 'rop_envs.envs:EckelEnvDisc')
register(id = 'eckel-mdisc-v0',
        entry_point = 'rop_envs.envs:EckelEnvMDisc')
register(id = 'rop-v0',
        entry_point = 'rop_envs.envs:EckelRate')