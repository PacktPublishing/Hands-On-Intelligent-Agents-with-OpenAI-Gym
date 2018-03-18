#!/usr/bin/env python

import gym
import sys

def run_env(argv):
    env = gym.make(argv[1])
    env.reset()
    for _ in range(int(argv[2])):
        env.render()
        env.step(env.action_space.sample())
    env.close()
    
if __name__ == "__main__":
    run_env(sys.argv)
