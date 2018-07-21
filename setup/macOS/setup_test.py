#!/usr/bin/env python
import gym
env = gym.make("LunarLander-v2")
# Run a sample/random agent for 10 episodes
for _ in range(10):
    _ = env.reset()
    env.render()
    done = False
    while not done:
        _, _, done, _ = env.step(env.action_space.sample())
        env.render()
