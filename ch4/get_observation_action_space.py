#!/usr/bin/env python
# Handy script for exploring Gym environment's spaces | Praveen Palanisamy
# Chapter 4, Hands-on Intelligent Agents with OpenAI Gym, 2018

import sys
import gym
from gym.spaces import *

def print_spaces(space):
    print(space)
    if isinstance(space, Box): # Print lower and upper bound if it's a Box space
        print("\n space.low: ", space.low)
        print("\n space.high: ", space.high)

    
if __name__ == "__main__":
    env = gym.make(sys.argv[1])
    print("Observation Space:")
    print_spaces(env.observation_space)
    print("Action Space:")
    print_spaces(env.action_space)
    try:
        print("Action description/meaning:",env.unwrapped.get_action_meanings())
    except AttributeError:
        pass
