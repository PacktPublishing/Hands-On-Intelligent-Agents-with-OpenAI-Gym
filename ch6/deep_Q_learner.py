#!/usr/bin/env python
import gym
import random
import torch
from collections import namedtuple
from torch.autograd import Variable
import numpy as np
from decay_schedule import LinearDecaySchedule
from utils.experience_memory import Experience, ExperienceMemory
from tensorboardX import SummaryWriter
from datetime import datetime

env = gym.make("CartPole-v0")
MAX_NUM_EPISODES = 70000
MAX_STEPS_PER_EPISODE = 300
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

class NN1(torch.nn.Module):
    def __init__(self, input_shape, output_shape, hidden_shape=40, seed=SEED):

        super(NN1, self).__init__()
        self.linear1 = torch.nn.Linear(input_shape, hidden_shape)
        self.out = torch.nn.Linear(hidden_shape, output_shape)

    def forward(self, x):
        x = Variable(torch.from_numpy(x)).float().to(device)
        x = torch.nn.functional.relu(self.linear1(x))
        x = self.out(x)
        return x


class Deep_Q_Learner(object):
    def __init__(self, state_shape, action_shape, learning_rate=0.005,
                 gamma=0.98):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.gamma = gamma  # Agent's discount factor
        self.learning_rate = learning_rate  # Agent's Q-learning rate
        # self.Q is the Action-Value function. This agent represents Q using a
        # Neural Network.
        self.Q = NN1(state_shape, action_shape).to(device)
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=1e-3)
        if USE_TARGET_NETWORK:
            self.Q_target = NN1(state_shape, action_shape).to(device)
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
    observation_shape = env.observation_space.shape[0]
    action_shape = env.action_space.n
    agent = Deep_Q_Learner(observation_shape, action_shape)
    first_episode = True
    episode_rewards = list()
    for episode in range(MAX_NUM_EPISODES):
        obs = env.reset()
        cum_reward = 0.0  # Cumulative reward
        for step in range(MAX_STEPS_PER_EPISODE):
            # env.render()
            action = agent.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            #agent.learn(obs, action, reward, next_obs, done)
            agent.memory.store(Experience(obs, action, reward, next_obs, done))

            obs = next_obs
            cum_reward += reward
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
