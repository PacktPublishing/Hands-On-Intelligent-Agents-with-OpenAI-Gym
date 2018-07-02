#!/usr/bin/env/ python
from OpenGL import GLU  # Temporary fix for roboschool issue #8
import roboschool
import gym
from argparse import ArgumentParser

argparser = ArgumentParser()
argparser.add_argument("--env",
                       help="Roboschool environment name. Default:RoboschoolInvertedPendulum-v1",
                       default="RoboschoolInvertedPendulum-v1")
args = argparser.parse_args()

env = gym.make(args.env)
obs = env.reset()
env = gym.wrappers.Monitor(env, "./roboschool_clips/" + args.env,
                           video_callable=lambda episode_id: True,
                           force=True)  #Rewrite prev recorded files if present
print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space)

for episode in range(3):  # Run 3 episode
    done = False
    obs = env.reset()
    while not done:  # So that Monitor can record at least 3 episodes
        _, _, done, _ = env.step(env.action_space.sample())
        #env.render()
