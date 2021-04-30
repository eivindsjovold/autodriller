from gym.envs.registration import register

register(id = 'by-v0',
        entry_point = 'rop_envs.envs:ByModEnv')
register(id = 'eckel-outdated-v0',
        entry_point = 'rop_envs.envs:EckelEnv')
register(id = 'eckel-disc-outdated-v0',
        entry_point = 'rop_envs.envs:EckelEnvDisc')
register(id = 'eckel-mdisc-outdated-v0',
        entry_point = 'rop_envs.envs:EckelEnvMDisc')
register(id = 'rop-outdated-v0',
        entry_point = 'rop_envs.envs:EckelRate')
register(id = 'rop-v1',
        entry_point = 'rop_envs.envs:BYRate')
register(id = 'bm-v0',
        entry_point = 'rop_envs.envs:BenchmarkEckel')
register(id = 'rop-iv-v0',
        entry_point = 'rop_envs.envs:EckelRateIV')

register(id = 'eckel-v0',
        entry_point = 'rop_envs.envs:EckelEnv1')

register(id = 'eckel-v1',
        entry_point = 'rop_envs.envs:EckelEnv2')
register(id = 'eckel-test-v0',
        entry_point = 'rop_envs.envs:EckelTestEnv1')
register(id = 'eckel-test-v1',
        entry_point = 'rop_envs.envs:EckelTestEnv2')
register(id = 'by-v1',
        entry_point = 'rop_envs.envs:BYEnv')
register(id = 'by-test-v1',
        entry_point = 'rop_envs.envs:BYTestEnv')

register(id = 'by-cont-v1',
        entry_point = 'rop_envs.envs:BYContEnv')
register(id = 'eckel-cont-v0',
        entry_point = 'rop_envs.envs:EckelContEnv1')
register(id = 'eckel-cont-v1',
        entry_point = 'rop_envs.envs:EckelContEnv2')

register(id = 'simple-v0',
        entry_point = 'rop_envs.envs:SimpleEnv1')
register(id = 'simple-v1',
        entry_point = 'rop_envs.envs:SimpleEnv2')
register(id = 'simple-v2',
        entry_point = 'rop_envs.envs:SimpleEnv3')


register(id = 'memory-eckel-v0',
        entry_point = 'rop_envs.envs:EckelMemory1')


