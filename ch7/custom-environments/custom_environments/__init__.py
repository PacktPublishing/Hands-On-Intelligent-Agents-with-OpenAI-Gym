from gym.envs.registration import register

register(
    id='custom-v0',
    entry_point='custom_environments.envs:CustomEnv',
)