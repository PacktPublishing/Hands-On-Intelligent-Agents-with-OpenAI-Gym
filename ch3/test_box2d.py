#!/usr/bin/env python
# Simple script to test a box2d environment | Praveen Palanisamy
# Chapter 3, Hands-on Intelligent Agents with OpenAI Gym, 2018

import gym
env = gym.make('BipedalWalker-v2')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())
