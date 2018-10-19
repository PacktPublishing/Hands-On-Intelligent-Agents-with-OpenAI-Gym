#!/usr/bin/env python
# Handy script for listing all available Gym environments | Praveen Palanisamy
# Chapter 4, Hands-on Intelligent Agents with OpenAI Gym, 2018

from gym import envs
env_names = [spec.id for spec in envs.registry.all()]
for name in sorted(env_names):
    print(name)
