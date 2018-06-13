#!/usr/bin/env python
import numpy as np
import torch
import gym
from argparse import ArgumentParser

parser = ArgumentParser("deep_ac_agent")
parser.add_argument("--env-name",
                    type= str,
                    default="CarRacing-v0",
                    metavar="ENV_ID")
args = parser.parse_args()
global_step_num = 0

# Introduced in PyTorch 0.4
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
torch.manual_seed(seed)
np.random.seed(seed)


if __name__ == "__main__":
    env = gym.make(args.env_name)
    observation_shape = env.observation_space.shape
    action_shape = env.action_space.shape[0]
    agent = Deep_AC_Agent(observation_shape, action_shape, agent_params)

    for episode in range(agent_params["max_num_episodes"]):
        obs = env.reset()
        done = False
        ep_reward = 0
        step_num = 0
        while not done:
            action = agent.get_action(obs).numpy()
            next_obs, reward, done, info = env.step(action)
            agent.learn_td_ac(obs, action, reward, next_obs, done)

            obs = next_obs
            ep_reward += reward
            step_num += 1
            global_step_num += 1


