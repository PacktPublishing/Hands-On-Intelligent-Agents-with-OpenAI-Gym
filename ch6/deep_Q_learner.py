#!/usr/bin/env python

import gym
import torch
import random
import numpy as np

from utils.params_manager import ParamsManager
from utils.decay_schedule import LinearDecaySchedule
from utils.experience_memory import Experience, ExperienceMemory
from function_approximator.perceptron import SLP
from function_approximator.cnn import CNN
from tensorboardX import SummaryWriter
from datetime import datetime
from argparse import ArgumentParser

args = ArgumentParser("Deep-Q-Learner arguments")
args.add_argument("--params-file",
                  help="Path to the parameters json file. Default is parameters.json",
                  default="parameters.json",
                  type=str,
                  metavar="F")
args.add_argument("--env-name",
                  help="ID of the Atari environment available in OpenAI Gym",
                  default="CartPole-v0",
                  type=str,
                  metavar="E")
args = args.parse_args()

params_manager= ParamsManager(args.params_file)
ENV_NAME = args.env_name
SEED = params_manager.get_agent_params()['seed']
summary_file_name = params_manager.get_agent_params()['summary_filename_prefix'] + datetime.now().strftime("%y-%m-%d-%H-%M")
writer = SummaryWriter(summary_file_name)
global_step_num = 0
# new in PyTorch 0.4
device = torch.device("cuda" if torch.cuda.is_available() and params_manager.get_agent_params()['use_cuda'] else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available() and USE_CUDA:
    torch.cuda.manual_seed_all(SEED)


class Deep_Q_Learner(object):
    def __init__(self, state_shape, action_shape, params):
        """
        self.Q is the Action-Value function. This agent represents Q using a Neural Network
        If the input is a single dimensional vector, uses a Single-Layer-Perceptron else if the input is 3 dimensional
        image, use a Convolutional-Neural-Network

        :param state_shape: Shape (tuple) of the observation/state
        :param action_shape: Shape (number) of the discrete action space
        :param params: A dictionary containing various Agent configuration parameters and hyper-parameters
        """
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.params = params
        self.gamma = self.params['gamma']  # Agent's discount factor
        self.learning_rate = self.params['lr']  # Agent's Q-learning rate

        if len(self.state_shape) == 1:  # Single dimensional observation/state space
            self.DQN = SLP
        elif len(self.state_shape) == 3:  # 3D/image observation/state
            self.DQN = CNN

        self.Q = self.DQN(state_shape, action_shape, device).to(device)
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=self.learning_rate)
        if self.params['use_target_network']:
            self.Q_target = self.DQN(state_shape, action_shape, device).to(device)
        # self.policy is the policy followed by the agent. This agents follows
        # an epsilon-greedy policy w.r.t it's Q estimate.
        self.policy = self.epsilon_greedy_Q
        self.epsilon_max = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = LinearDecaySchedule(initial_value=self.epsilon_max,
                                    final_value=self.epsilon_min,
                                    max_steps= 0.1 * self.params['max_num_episodes'] * \
                                               self.params['max_steps_per_episode'])
        self.step_num = 0
                
        self.memory = ExperienceMemory(capacity=int(1e6))  # Initialize an Experience memory with 1M capacity

    def get_action(self, observation):
        if len(observation.shape) == 3: # Single image (not a batch)
            if observation.shape[2] < observation.shape[0]:  # Probably observation is in W x H x C format
                # Reshape to C x H x W format as per PyTorch's convention
                observation = observation.reshape(observation.shape[2], observation.shape[1], observation.shape[0])
            observation = np.expand_dims(observation, 0)  # Create a batch dimension
        return self.policy(observation)

    def epsilon_greedy_Q(self, observation):
        # Decay Epsilon/exploration as per schedule
        writer.add_scalar("DQL/epsilon", self.epsilon_decay(self.step_num), self.step_num)
        self.step_num +=1
        if random.random() < self.epsilon_decay(self.step_num):
            action = random.choice([i for i in range(self.action_shape)])
        else:
            action = np.argmax(self.Q(observation).data.to(torch.device('cpu')).numpy())


        return action

    def learn(self, s, a, r, s_next, done):
        # TD(0) Q-learning
        if done:  # End of episode
            td_target = reward + 0.0  # Set the value of terminal state to zero
        else:
            td_target = r + self.gamma * torch.max(self.Q(s_next))
        td_error = td_target - self.Q(s)[a]
        # Update Q estimate
        #self.Q(s)[a] = self.Q(s)[a] + self.learning_rate * td_error
        self.Q_optimizer.zero_grad()
        td_error.backward()
        self.Q_optimizer.step()

    def learn_from_batch_experience(self, experience):
        batch_xp = Experience(*zip(*experience))
        obs_batch = np.array(batch_xp.obs)
        action_batch = np.array(batch_xp.action)
        reward_batch = np.array(batch_xp.reward)
        next_obs_batch = np.array(batch_xp.next_obs)
        done_batch = np.array(batch_xp.done)

        if self.params['use_target_network']:
            if self.step_num % self.params['target_network_update_freq'] == 0:
                # The *update_freq is the Num steps after which target net is updated.
                # A schedule can be used instead to vary the update freq.
                self.Q_target.load_state_dict(self.Q.state_dict())
            td_target = reward_batch + ~done_batch * \
                np.tile(self.gamma, len(next_obs_batch)) * \
                self.Q_target(next_obs_batch).max(1)[0].data
        else:
            td_target = reward_batch + ~done_batch * \
                np.tile(self.gamma, len(next_obs_batch)) * \
                self.Q(next_obs_batch).detach().max(1)[0].data

        td_target = td_target.to(device)
        action_idx = torch.from_numpy(action_batch).to(device)
        td_error = torch.nn.functional.mse_loss( self.Q(obs_batch).gather(1, action_idx.view(-1, 1)),
                                                       td_target.float().unsqueeze(1))

        self.Q_optimizer.zero_grad()
        td_error.mean().backward()
        writer.add_scalar("DQL/td_error", td_error.mean(), self.step_num)
        self.Q_optimizer.step()

    def replay_experience(self, batch_size = None):
        batch_size = batch_size if batch_size is not None else self.params['replay_batch_size']
        experience_batch = self.memory.sample(batch_size)
        self.learn_from_batch_experience(experience_batch)

if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    observation_shape = env.observation_space.shape
    action_shape = env.action_space.n
    agent_params = params_manager.get_agent_params()
    agent = Deep_Q_Learner(observation_shape, action_shape, agent_params)
    first_episode = True
    episode_rewards = list()
    for episode in range(agent_params['max_num_episodes']):
        obs = env.reset()
        cum_reward = 0.0  # Cumulative reward
        done = False
        step = 0
        #for step in range(agent_params['max_steps_per_episode']):
        while not done:
            #env.render()
            action = agent.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            #agent.learn(obs, action, reward, next_obs, done)
            agent.memory.store(Experience(obs, action, reward, next_obs, done))

            obs = next_obs
            cum_reward += reward
            step += 1
            global_step_num +=1

            if done is True:
                if first_episode:  # Initialize max_reward at the end of first episode
                    max_reward = cum_reward
                    first_episode = False
                episode_rewards.append(cum_reward)
                if cum_reward > max_reward:
                    max_reward = cum_reward
                print("\nEpisode#{} ended in {} steps. reward ={} ; mean_reward={:.3f} best_reward={}".
                      format(episode, step+1, cum_reward, np.mean(episode_rewards), max_reward))
                writer.add_scalar("main/ep_reward", cum_reward, episode)
                writer.add_scalar("main/mean_ep_reward", np.mean(episode_rewards), episode)
                writer.add_scalar("main/max_ep_rew", max_reward, episode)

                if agent.memory.get_size() >= 2 * agent_params['replay_batch_size']:
                    agent.replay_experience()
                break
    env.close()
    writer.close()
