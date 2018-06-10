from gym.envs.registration import register

register(
    id='Carla-v0',
    entry_point='carla_gym.envs:CarlaEnv',
)