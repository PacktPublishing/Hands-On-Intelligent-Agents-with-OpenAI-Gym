from gym.envs.registration import register

register(
    id='Carla-v0',
    entry_point='environment.carla_gym.envs:CarlaEnv',
)