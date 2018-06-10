from gym.envs.registration import register

register(
    id='CustomEnv-v0',
    entry_point='custom_environments.envs:CustomEnv',
)