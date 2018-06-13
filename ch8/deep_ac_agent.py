#!/usr/bin/env python
import numpy as np
import torch
import gym
from argparse import ArgumentParser
from utils.params_manager import ParamsManager

parser = ArgumentParser("deep_ac_agent")
parser.add_argument("--env-name",
                    type= str,
                    default="CarRacing-v0",
                    metavar="ENV_ID")
parser.add_argument("--params-file",
                    type= str,
                    default="parameters.json",
                    metavar="PFILE.json")
args = parser.parse_args()
global_step_num = 0

params_manager= ParamsManager(args.params_file)
seed = params_manager.get_agent_params()['seed']
# Export the parameters as json files to the log directory to keep track of the parameters used in each experiment
params_manager.export_env_params(summary_file_path + "/" + "env_params.json")
params_manager.export_agent_params(summary_file_path + "/" + "agent_params.json")
use_cuda = params_manager.get_agent_params()['use_cuda']
# Introduced in PyTorch 0.4
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available() and use_cuda:
    torch.cuda.manual_seed_all(seed)


class Deep_AC(torch.nn.Module):
    def __init__(self, input_shape, actor_shape, critic_shape, params):
        super(Deep_AC, self).__init__()
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(input_shape[2], 128, 3, stride=1, padding=0),
                                          torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Conv2d(128, 64, 3, stride=1, padding=0),
                                          torch.nn.ReLU())
        self.layer3 = torch.nn.Sequential(torch.nn.Conv2d(64, 32, 3, stride=1, padding=0),
                                          torch.nn.ReLU())
        self.layer4 = torch.nn.Sequential(torch.nn.Linear(32 * 78 * 78, 2048),
                                          torch.nn.ReLU())
        self.actor_mu = torch.nn.Linear(2048, actor_shape)
        self.actor_sigma = torch.nn.Linear(2048, actor_shape)
        self.critic = torch.nn.Linear(2048, critic_shape)

    def forward(self, x):
        x.requires_grad_()
        x = x.cuda()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.shape[0], -1)
        x = self.layer4(x)
        actor_mu = self.actor_mu(x)
        actor_sigma = self.actor_sigma(x)
        critic = self.critic(x)
        return actor_mu, actor_sigma, critic



if __name__ == "__main__":
    env = gym.make(args.env_name)
    observation_shape = env.observation_space.shape
    action_shape = env.action_space.shape[0]
    agent_params = params_manager.get_agent_params()
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
            #env.render()
            print("Episode#:", episode, "step#:", step_num, "\t rew=", reward, end="\r")
        print("Episode#:", episode, "\t ep_reward=", ep_reward)


