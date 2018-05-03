#!/usr/bin/env python

import gym
import torch
from torch.autograd import Variable
import random
import numpy as np

from decay_schedule import LinearDecaySchedule
from utils.experience_memory import Experience, ExperienceMemory
from tensorboardX import SummaryWriter
from datetime import datetime

ENV_NAME = "CartPole-v0"
MAX_NUM_EPISODES = 70000
MAX_STEPS_PER_EPISODE = 300  # Currently only used for calculating epsilon decay schedule
REPLAY_BATCH_SIZE = 2000
SEED = 555
USE_CUDA = True
USE_TARGET_NETWORK = True
# Num steps after which target net is updated. A schedule can be used instead to vary the update freq
TARGET_NETWORK_UPDATE_FREQ = 2000
summary_file_name = "logs/DQL" + datetime.now().strftime("%y-%m-%d-%H-%M")
writer = SummaryWriter(summary_file_name)
global_step_num = 0

device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")  # new in PyTorch 0.4
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available() and USE_CUDA:
    torch.cuda.manual_seed_all(SEED)

class SLP(torch.nn.Module):
    """
    A Single Layer Perceptron (SLP) class to approximate functions
    """
    def __init__(self, input_shape, output_shape):
        """
        :param input_shape: Shape/dimension of the input
        :param output_shape: Shape/dimension of the output
        """
        super(SLP, self).__init__()
        self.input_shape = input_shape[0]
        self.hidden_shape = 40
        self.linear1 = torch.nn.Linear(self.input_shape, self.hidden_shape)
        self.out = torch.nn.Linear(self.hidden_shape, output_shape)

    def forward(self, x):
        x = Variable(torch.from_numpy(x)).float().to(device)
        x = torch.nn.functional.relu(self.linear1(x))
        x = self.out(x)
        return x


class CNN(torch.nn.Module):
    """
    A Convolution Neural Network (CNN) class to approximate functions with visual/image inputs
    """
    def __init__(self, input_shape, output_shape):
        """
        :param input_shape:  Shape/dimension of the input image. Assumed to be resized to C x 84 x 84
        :param output_shape: Shape/dimension of the output.
        """
        #  input_shape: C x 84 x 84
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_shape[0], 64, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=0),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU()
        )
        self.out = torch.nn.Linear(18 * 18 * 32, output_shape)

    def forward(self, x):
        x = torch.from_numpy(x).float().to(device)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.shape[0], -1)
        x = self.out(x)
        return x


class Deep_Q_Learner(object):
    def __init__(self, state_shape, action_shape, learning_rate=0.005,
                 gamma=0.98):
        """
        self.Q is the Action-Value function. This agent represents Q using a Neural Network
        If the input is a single dimensional vector, uses a Single-Layer-Perceptron else if the input is 3 dimensional
        image, use a Convolutional-Neural-Network

        :param state_shape: Shape (tuple) of the observation/state
        :param action_shape: Shape (number) of the discrete action space
        :param learning_rate: Agent's (Q-)Learning rate for the Neural Network (default=0.005)
        :param gamma: Agent's Discount factor (default=0.98)
        """
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.gamma = gamma  # Agent's discount factor
        self.learning_rate = learning_rate  # Agent's Q-learning rate

        if len(self.state_shape) == 1:  # Single dimensional observation/state space
            self.DQN = SLP
        elif len(self.state_shape) == 3:  # 3D/image observation/state
            self.DQN = CNN

        self.Q = self.DQN(state_shape, action_shape).to(device)
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=1e-3)
        if USE_TARGET_NETWORK:
            self.Q_target = self.DQN(state_shape, action_shape).to(device)
        # self.policy is the policy followed by the agent. This agents follows
        # an epsilon-greedy policy w.r.t it's Q estimate.
        self.policy = self.epsilon_greedy_Q
        self.epsilon_max = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = LinearDecaySchedule(initial_value=self.epsilon_max,
                                    final_value=self.epsilon_min,
                                    max_steps= 0.1 * MAX_NUM_EPISODES * MAX_STEPS_PER_EPISODE)
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

        if USE_TARGET_NETWORK:
            if self.step_num % TARGET_NETWORK_UPDATE_FREQ == 0:
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

    def replay_experience(self, batch_size=REPLAY_BATCH_SIZE):
        experience_batch = self.memory.sample(batch_size)
        self.learn_from_batch_experience(experience_batch)

if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    observation_shape = env.observation_space.shape
    action_shape = env.action_space.n
    agent = Deep_Q_Learner(observation_shape, action_shape)
    first_episode = True
    episode_rewards = list()
    for episode in range(MAX_NUM_EPISODES):
        obs = env.reset()
        cum_reward = 0.0  # Cumulative reward
        done = False
        step = 0
        #for step in range(MAX_STEPS_PER_EPISODE):
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

                if agent.memory.get_size() >= 2 * REPLAY_BATCH_SIZE:
                    agent.replay_experience()
                break
    env.close()
    writer.close()
