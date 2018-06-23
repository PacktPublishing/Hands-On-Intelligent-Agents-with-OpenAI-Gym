#!/usr/bin/env python
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.multiprocessing as mp
import torch.nn.functional as F
import gym
try:
    import roboschool
except ImportError:
    pass
from argparse import ArgumentParser
from datetime import datetime
from collections import namedtuple
from tensorboardX import SummaryWriter
from utils.params_manager import ParamsManager
from function_approximator.shallow import Actor as ShallowActor
from function_approximator.shallow import Critic as ShallowCritic
from function_approximator.deep import Actor as DeepActor
from function_approximator.deep import Critic as DeepCritic

parser = ArgumentParser("deep_ac_agent")
parser.add_argument("--env", help="Name of the Gym environment",
                    type= str,
                    default="CarRacing-v0",
                    metavar="ENV_ID")
parser.add_argument("--params-file", help="Path to the parameters file. Default= ./parameters.json",
                    type= str,
                    default="parameters.json",
                    metavar="PFILE.json")
parser.add_argument("--model-dir", help="Directory to save/load trained model. Default= ./trained_models/",
                    type=str,
                    default="trained_models/",
                    metavar="MODEL_DIR")
args = parser.parse_args()
global_step_num = 0

params_manager= ParamsManager(args.params_file)
summary_file_path_prefix = params_manager.get_agent_params()['summary_file_path_prefix']
summary_file_path= summary_file_path_prefix + args.env + "_" + datetime.now().strftime("%y-%m-%d-%H-%M")
writer = SummaryWriter(summary_file_path)
# Export the parameters as json files to the log directory to keep track of the parameters used in each experiment
params_manager.export_env_params(summary_file_path + "/" + "env_params.json")
params_manager.export_agent_params(summary_file_path + "/" + "agent_params.json")
use_cuda = params_manager.get_agent_params()['use_cuda']
# Introduced in PyTorch 0.4
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

seed = params_manager.get_agent_params()['seed']  # With the intent to make the results reproducible
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available() and use_cuda:
    torch.cuda.manual_seed_all(seed)

Transition = namedtuple("Transition", ["s", "value_s", "a", "log_prob_a"])

class DeepActorCriticAgent(mp.Process):
    def __init__(self, id, env_name, agent_params):
        """
        An Actor-Critic Agent that uses a Deep Neural Network to represent it's Policy and the Value function
        :param state_shape:
        :param action_shape:
        """
        super(DeepActorCriticAgent, self).__init__()
        self.id = id
        self.actor_name = "actor" + str(self.id)
        self.env_name = env_name
        self.params = agent_params
        self.policy = self.multi_variate_gaussian_policy
        self.gamma = self.params['gamma']
        self.trajectory = []  # Contains the trajectory of the agent as a sequence of Transitions
        self.rewards = []  #  Contains the rewards obtained from the env at every step
        self.global_step_num = 0
        self.best_mean_reward = - float("inf") # Agent's personal best mean episode reward
        self.best_reward = - float("inf")
        self.saved_params = False  # Whether or not the params have been saved along with the model to model_dir

    def multi_variate_gaussian_policy(self, obs):
        """
        Calculates a multi-variate gaussian distribution over actions given observations
        :param obs: Agent's observation
        :return: policy, a distribution over actions for the given observation
        """
        mu, sigma = self.actor(obs)
        value = self.critic(obs)
        [ mu[:, i].clamp_(float(self.env.action_space.low[i]), float(self.env.action_space.high[i]))
        for i in range(self.action_shape)]  # Clamp each dim of mu based on the (low,high) limits of that action dim
        sigma = torch.nn.Softplus()(sigma).squeeze() + 1e-7  # Let sigma be (smoothly) +ve
        self.mu = mu.to(torch.device("cpu"))
        self.sigma = sigma.to(torch.device("cpu"))
        self.value = value.to(torch.device("cpu"))
        if len(self.mu.shape) == 0: # See if mu is a scalar
            #self.mu = self.mu.unsqueeze(0)  # This prevents MultivariateNormal from crashing with SIGFPE
            self.mu.unsqueeze_(0)
        self.action_distribution = MultivariateNormal(self.mu, torch.eye(self.action_shape) * self.sigma, validate_args=True)
        return(self.action_distribution)

    def preproc_obs(self, obs):
        if len(obs.shape) == 3:
            #  Make sure the obs are in this order: C x W x H and add a batch dimension
            obs = np.reshape(obs, (obs.shape[2], obs.shape[1], obs.shape[0]))
            obs = np.resize(obs, (3, 84, 84))
        #  Convert to torch Tensor, add a batch dimension, convert to float repr
        obs = torch.from_numpy(obs).unsqueeze(0).float()
        return obs

    def process_action(self, action):
        [action[:, i].clamp_(float(self.env.action_space.low[i]), float(self.env.action_space.high[i]))
         for i in range(self.action_shape)]  # Limit the action to lie between the (low, high) limits of the env
        action = action.to(torch.device("cpu"))
        return action.numpy().squeeze(0)  # Convert to numpy ndarray, squeeze and remove the batch dimension

    def get_action(self, obs):
        obs = self.preproc_obs(obs)
        action_distribution = self.policy(obs)  # Call to self.policy(obs) also populates self.value with V(obs)
        value = self.value
        action = action_distribution.sample()
        log_prob_a = action_distribution.log_prob(action)
        action = self.process_action(action)
        self.trajectory.append(Transition(obs, value, action, log_prob_a))  # Construct the trajectory
        return action

    def calculate_n_step_return(self, n_step_rewards, final_state, done, gamma):
        """
        Calculates the n-step return for each state in the input-trajectory/n_step_transitions
        :param n_step_rewards: List of rewards for each step
        :param final_state: Final state in this n_step_transition/trajectory
        :param done: True rf the final state is a terminal state if not, False
        :return: The n-step return for each state in the n_step_transitions
        """
        g_t_n_s = list()
        with torch.no_grad():
            g_t_n = torch.tensor([[0]]).float() if done else self.critic(self.preproc_obs(final_state)).cpu()
            for r_t in n_step_rewards[::-1]:  # Reverse order; From r_tpn to r_t
                g_t_n = torch.tensor(r_t).float() + self.gamma * g_t_n
                g_t_n_s.insert(0, g_t_n)  # n-step returns inserted to the left to maintain correct index order
            return g_t_n_s

    def calculate_loss(self, trajectory, td_targets):
        """
        Calculates the critic and actor losses using the td_targets and self.trajectory
        :param td_targets:
        :return:
        """
        n_step_trajectory = Transition(*zip(*trajectory))
        v_s_batch = n_step_trajectory.value_s
        log_prob_a_batch = n_step_trajectory.log_prob_a
        actor_loss, critic_loss = [], []
        for td_target, critic_prediction, log_p_a in zip(td_targets, v_s_batch, log_prob_a_batch):
            td_err = td_target - critic_prediction
            actor_loss.append(- log_p_a * td_err)  # td_err is an unbiased estimated of Advantage
            critic_loss.append(F.smooth_l1_loss(critic_prediction, td_target))
            #critic_loss.append(F.mse_loss(critic_pred, td_target))
        actor_loss = torch.stack(actor_loss).mean()
        critic_loss = torch.stack(critic_loss).mean()

        writer.add_scalar(self.actor_name + "/critic_loss", critic_loss, self.global_step_num)
        writer.add_scalar(self.actor_name + "/actor_loss", actor_loss, self.global_step_num)

        return actor_loss, critic_loss

    def learn(self, n_th_observation, done):
        td_targets = self.calculate_n_step_return(self.rewards, n_th_observation, done, self.gamma)
        actor_loss, critic_loss = self.calculate_loss(self.trajectory, td_targets)

        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.trajectory.clear()
        self.rewards.clear()

    def save(self):
        model_file_name = self.params["model_dir"] + "A2C_" + self.env_name + ".ptm"
        agent_state = {"Actor": self.actor.state_dict(),
                       "Critic": self.critic.state_dict(),
                       "best_mean_reward": self.best_mean_reward,
                       "best_reward": self.best_reward}
        torch.save(agent_state, model_file_name)
        print("Agent's state is saved to", model_file_name)
        # Export the params used if not exported already
        if not self.saved_params:
            params_manager.export_agent_params(model_file_name + ".agent_params")
            print("The parameters have been saved to", model_file_name + ".agent_params")
            self.saved_params = True

    def load(self):
        model_file_name = self.params["model_dir"] + "A2C_" + self.env_name + ".ptm"
        agent_state = torch.load(model_file_name, map_location= lambda storage, loc: storage)
        self.actor.load_state_dict(agent_state["Actor"])
        self.critic.load_state_dict(agent_state["Critic"])
        self.actor.to(device)
        self.critic.to(device)
        self.best_mean_reward = agent_state["best_mean_reward"]
        self.best_reward = agent_state["best_reward"]
        print("Loaded Advantage Actor-Critic model state from", model_file_name,
              " which fetched a best mean reward of:", self.best_mean_reward,
              " and an all time best reward of:", self.best_reward)

    def run(self):
        self.env = gym.make(self.env_name)
        self.state_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.shape[0]
        self.critic_shape = 1
        if len(self.state_shape) == 3:  # Screen image is the input to the agent
            self.actor= DeepActor(self.state_shape, self.action_shape, device).to(device)
            self.critic = DeepCritic(self.state_shape, self.critic_shape, device).to(device)
        else:  # Input is a (single dimensional) vector
            #self.actor_critic = ShallowActorCritic(self.state_shape, self.action_shape, 1, self.params).to(device)
            self.actor = ShallowActor(self.state_shape, self.action_shape, device).to(device)
            self.critic = ShallowCritic(self.state_shape, self.critic_shape, device).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

        # Handle loading and saving of trained Agent models
        episode_rewards = list()
        prev_checkpoint_mean_ep_rew = self.best_mean_reward
        num_improved_episodes_before_checkpoint = 0  # To keep track of the num of ep with higher perf to save model
        #print("Using agent_params:", self.params)
        if self.params['load_trained_model']:
            try:
                self.load()
                prev_checkpoint_mean_ep_rew = self.best_mean_reward
            except FileNotFoundError:
                print("WARNING: No trained model found for this environment. Training from scratch.")

        for episode in range(self.params["max_num_episodes"]):
            obs = self.env.reset()
            done = False
            ep_reward = 0.0
            step_num = 0
            while not done:
                action = self.get_action(obs)
                next_obs, reward, done, _ = self.env.step(action)
                self.rewards.append(reward)
                cum_reward = np.sum(self.rewards)
                step_num +=1
                if step_num >= self.params["learning_step_thresh"] or done:
                    self.learn(next_obs, done)
                    step_num = 0
                    # Monitor performance and save Agent's state when perf improves
                    if done:
                        episode_rewards.append(cum_reward)
                        if cum_reward > self.best_reward:
                            self.best_reward = cum_reward
                        if np.mean(episode_rewards) > prev_checkpoint_mean_ep_rew:
                            num_improved_episodes_before_checkpoint += 1
                        if num_improved_episodes_before_checkpoint >= self.params["save_freq_when_perf_improves"]:
                            prev_checkpoint_mean_ep_rew = np.mean(episode_rewards)
                            self.best_mean_reward = np.mean(episode_rewards)
                            self.save()
                            num_improved_episodes_before_checkpoint = 0

                obs = next_obs
                ep_reward += reward
                self.global_step_num += 1
                #print(self.actor_name + ":Episode#:", episode, "step#:", step_num, "\t rew=", reward, end="\r")
                writer.add_scalar(self.actor_name + "/reward", reward, self.global_step_num)
            print("{}:Episode#:{} \t ep_reward:{} \t best_ep_reward:{}".format(
                self.actor_name, episode, ep_reward, self.best_reward))
            writer.add_scalar(self.actor_name + "/ep_reward", ep_reward, self.global_step_num)


if __name__ == "__main__":
    agent_params = params_manager.get_agent_params()
    agent_params["model_dir"] = args.model_dir
    mp.set_start_method('spawn')

    agent_procs =[DeepActorCriticAgent(id, args.env, agent_params) for id in range(agent_params["num_agents"])]
    [p.start() for p in agent_procs]
    [p.join() for p in agent_procs]
