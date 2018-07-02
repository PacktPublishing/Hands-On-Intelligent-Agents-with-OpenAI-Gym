#!/usr/bin/env python
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
import torch.multiprocessing as mp
import torch.nn.functional as F
from environment.utils import SubprocVecEnv
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
from function_approximator.shallow import DiscreteActor as ShallowDiscreteActor
from function_approximator.shallow import Critic as ShallowCritic
from function_approximator.deep import Actor as DeepActor
from function_approximator.deep import DiscreteActor as DeepDiscreteActor
from function_approximator.deep import Critic as DeepCritic

parser = ArgumentParser("deep_ac_agent")
parser.add_argument("--env", help="Name of the Gym environment", default="CarRacing-v0", metavar="ENV_ID")
parser.add_argument("--params-file", help="Path to the parameters file. Default= ./parameters.json",
                    default="parameters.json", metavar="PFILE.json")
parser.add_argument("--model-dir", help="Directory to save/load trained model. Default= ./trained_models/",
                    default="trained_models/", metavar="MODEL_DIR")
parser.add_argument("--render", help="Whether to render the environment to the display. Default=False",
                    action='store_true', default=False)
parser.add_argument("--test", help="Tests a saved Agent model to see the performance. Disables learning",
                    action='store_true', default=False)
parser.add_argument("--gpu-id", help="GPU device ID to use. Default:0", type=int, default=0, metavar="GPU_ID")
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
device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() and use_cuda else "cpu")

seed = params_manager.get_agent_params()['seed']  # With the intent to make the results reproducible
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available() and use_cuda:
    torch.cuda.manual_seed_all(seed)

Transition = namedtuple("Transition", ["s", "value_s", "a", "log_prob_a"])


class DeepActorCriticAgent():
    def __init__(self, id, env_names, agent_params):
        """
        An Actor-Critic Agent that uses a Deep Neural Network to represent it's Policy and the Value function
        :param state_shape:
        :param action_shape:
        """
        super(DeepActorCriticAgent, self).__init__()
        self.id = id
        self.actor_name = "actor" + str(self.id)
        self.env_names = env_names
        self.params = agent_params
        self.policy = self.multi_variate_gaussian_policy
        self.gamma = self.params['gamma']
        self.trajectory = []  # Contains the trajectory of the agent as a sequence of Transitions
        self.rewards = []  #  Contains the rewards obtained from the env at every step
        self.global_step_num = 0
        self.best_mean_reward = - float("inf") # Agent's personal best mean episode reward
        self.best_reward = - float("inf")
        self.saved_params = False  # Whether or not the params have been saved along with the model to model_dir
        self.continuous_action_space = True  # Assumption by default unless env.action_space is Discrete

    def multi_variate_gaussian_policy(self, obs):
        """
        Calculates a multi-variate gaussian distribution over actions given observations
        :param obs: Agent's observation
        :return: policy, a distribution over actions for the given observation
        """
        mu, sigma = self.actor(obs)
        value = self.critic(obs).squeeze()
        [ mu[:, i].clamp_(float(self.envs.action_space.low[i]), float(self.envs.action_space.high[i]))
         for i in range(self.action_shape)]  # Clamp each dim of mu based on the (low,high) limits of that action dim
        sigma = torch.nn.Softplus()(sigma) + 1e-7  # Let sigma be (smoothly) +ve
        self.mu = mu.to(torch.device("cpu"))
        self.sigma = sigma.to(torch.device("cpu"))
        self.value = value.to(torch.device("cpu"))
        if len(self.mu[0].shape) == 0: # See if mu is a scalar
            self.mu = self.mu.unsqueeze(0)  # This prevents MultivariateNormal from crashing with SIGFPE
        self.covariance = torch.stack([torch.eye(self.action_shape) * s for s in self.sigma])
        if self.action_shape == 1:
            self.covariance = self.sigma.unsqueeze(-1)  # Make the covariance a square mat to avoid RuntimeError with MultivariateNormal
        self.action_distribution = MultivariateNormal(self.mu, self.covariance)
        return self.action_distribution

    def discrete_policy(self, obs):
        """
        Calculates a discrete/categorical distribution over actions given observations
        :param obs: Agent's observation
        :return: policy, a distribution over actions for the given observation
        """
        logits = self.actor(obs)
        value = self.critic(obs).squeeze()
        self.logits = logits.to(torch.device("cpu"))
        self.value = value.to(torch.device("cpu"))
        self.action_distribution = Categorical(logits=self.logits)
        return self.action_distribution

    def preproc_obs(self, obs):
        if len(obs[0].shape) == 3:  # shape of obs:(num_agents, obs_im_height, obs_im_width, obs_num_channels)
            #  Make sure the obs are in this order: C x W x H and add a batch dimension
            obs = np.reshape(obs, (-1, obs.shape[3], obs.shape[2], obs.shape[1]))
            #  The environment wrapper already takes care of reshaping image obs into 84 x 84 x C. Can be skipped
            obs = np.resize(obs, (-1, 3, 84, 84))
        #  Convert to torch Tensor, convert to float repr
        obs = torch.from_numpy(obs).float()
        return obs

    def process_action(self, action):
        if self.continuous_action_space:
            [action[:, i].clamp_(float(self.envs.action_space.low[i]), float(self.envs.action_space.high[i]))
             for i in range(self.action_shape)]  # Limit the action to lie between the (low, high) limits of the env
        action = action.to(torch.device("cpu"))
        return action.numpy()

    def get_action(self, obs):
        obs = self.preproc_obs(obs)
        action_distributions = self.policy(obs)  # Call to self.policy(obs) also populates self.value with V(obs)
        value = self.value
        actions = action_distributions.sample()
        log_prob_a = action_distributions.log_prob(actions)
        actions = self.process_action(actions)
        self.trajectory.append(Transition(obs, value, actions, log_prob_a))  # Construct the trajectory
        return actions
    # TODO: rename num_agents to num_actors in parameters.json file to be consistent with comments
    def calculate_n_step_return(self, n_step_rewards, next_states, dones, gamma):
        """
        Calculates the n-step return for each state in the input-trajectory/n_step_transitions for the "done" actors
        :param n_step_rewards: List of length=num_steps containing rewards of shape=(num_actors x 1)
        :param next_states: list of length=num_actors containing next observations of shape=(obs_shape)
        :param dones: list of length=num_actors containing True if the next_state is a terminal state if not, False
        :return: The n-step return for each state in the n_step_transitions
        """
        g_t_n_s = list()
        with torch.no_grad():
            # 1. Calculate next-state values for each actor:
            #    a. If next_state is terminal (done[actor_idx]=True), set g_t_n[actor_idx]=0
            #    b. If next_state is non-terminal (done[actor_idx]=False), set g_t_n[actor_idx] to Critic's prediction
            g_t_n = torch.tensor([[not d] for d  in dones]).float()  # 1. a.
            # See if there is at least one non-terminal next-state
            if np.where([not d for d in dones])[0].size > 0:
                non_terminal_idxs = torch.tensor(np.where([not d for d in dones])).squeeze(0)
                g_t_n[non_terminal_idxs] = self.critic(self.preproc_obs(next_states[non_terminal_idxs])).cpu()  # 1. b.
            g_t_n_s_batch = []
            n_step_rewards = torch.stack(n_step_rewards)  # tensor of shape (num_steps x num_actors x 1)
            # For each actor
            for actor_idx in range(n_step_rewards.shape[1]):
                actor_n_step_rewards = n_step_rewards.index_select(1, torch.tensor([actor_idx]))  # shape:(num_steps,1)
                g_t_n_s = []
                # Calculate n number of n-step returns
                for r_t in actor_n_step_rewards.numpy()[::-1]:  # Reverse order; From r_tpn to r_t; PyTorch can't slice in reverse #229
                    g_t_n[actor_idx] = torch.tensor(r_t).float() + self.gamma * g_t_n[actor_idx]
                    g_t_n_s.insert(0, g_t_n[actor_idx].clone())  # n-step returns inserted to the left to maintain correct index order
                g_t_n_s_batch.append(g_t_n_s)
            return torch.tensor(g_t_n_s_batch)  # tensor of shape:(num_actors, num_steps, 1)

    def calculate_loss(self, trajectory, td_targets):
        """
        Calculates the critic and actor losses using the td_targets and self.trajectory
        :param trajectory: List of trajectories from all the actors
        :param td_targets: Tensor of shape:(num_actors, num_steps, 1)
        :return:
        """
        n_step_trajectory = Transition(*zip(*trajectory))
        # n_step_trajectory.x returns a list of length= num_steps containing num_actors x shape_of_x items
        # 1. Create tensor of shape:(num_steps x num_actors x shape_of_x) (using torch.stack())
        # 2. Reshape the tensor to be of shape:(num_actors x num_steps x shape_of_x) (using torch.transpose(1,0)
        v_s_batch = torch.stack(n_step_trajectory.value_s).transpose(1, 0)  # shape:(num_actors, num_steps, 1)
        log_prob_a_batch = torch.stack(n_step_trajectory.log_prob_a).transpose(1, 0)  # shape:(num_actors, num_steps, 1)
        actor_losses, critic_losses = [], []
        for td_targets, critic_predictions, log_p_a in zip(td_targets, v_s_batch, log_prob_a_batch):
            td_err = td_targets - critic_predictions
            actor_losses.append(- log_p_a * td_err)  # td_err is an unbiased estimated of Advantage
            critic_losses.append(F.smooth_l1_loss(critic_predictions, td_targets))
            #critic_loss.append(F.mse_loss(critic_pred, td_target))
        if self.params["use_entropy_bonus"]:
            actor_loss = torch.stack(actor_losses).mean() - self.action_distribution.entropy().mean()
        else:
            actor_loss = torch.stack(actor_losses).mean()
        critic_loss = torch.stack(critic_losses).mean()

        writer.add_scalar(self.actor_name + "/critic_loss", critic_loss, self.global_step_num)
        writer.add_scalar(self.actor_name + "/actor_loss", actor_loss, self.global_step_num)

        return actor_loss, critic_loss

    def learn(self, n_th_observations, dones):
        td_targets = self.calculate_n_step_return(self.rewards, n_th_observations, dones, self.gamma)
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
        model_file_name = self.params["model_dir"] + "Batch-A2C_" + self.env_names[0] + ".ptm"
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
        model_file_name = self.params["model_dir"] + "Batch-A2C_" + self.env_names[0] + ".ptm"
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
        self.envs = SubprocVecEnv(self.env_names)
        self.state_shape = self.envs.observation_space.shape
        if isinstance(self.envs.action_space.sample(), int):  # Discrete action space
            self.action_shape = self.envs.action_space.n
            self.policy = self.discrete_policy
            self.continuous_action_space = False

        else:  # Continuous action space
            self.action_shape = self.envs.action_space.shape[0]
            self.policy = self.multi_variate_gaussian_policy
        self.critic_shape = 1
        if len(self.state_shape) == 3:  # Screen image is the input to the agent
            if self.continuous_action_space:
                self.actor= DeepActor(self.state_shape, self.action_shape, device).to(device)
            else:  # Discrete action space
                self.actor = DeepDiscreteActor(self.state_shape, self.action_shape, device).to(device)
            self.critic = DeepCritic(self.state_shape, self.critic_shape, device).to(device)
        else:  # Input is a (single dimensional) vector
            if self.continuous_action_space:
                #self.actor_critic = ShallowActorCritic(self.state_shape, self.action_shape, 1, self.params).to(device)
                self.actor = ShallowActor(self.state_shape, self.action_shape, device).to(device)
            else:  # Discrete action space
                self.actor = ShallowDiscreteActor(self.state_shape, self.action_shape, device).to(device)
            self.critic = ShallowCritic(self.state_shape, self.critic_shape, device).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.params["learning_rate"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.params["learning_rate"])

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
                if args.test:  # Test a saved model
                    print("FATAL: No saved model found. Cannot test. Press any key to train from scratch")
                    input()
                else:
                    print("WARNING: No trained model found for this environment. Training from scratch.")

        #for episode in range(self.params["max_num_episodes"]):
        obs = self.envs.reset()
        # TODO: Create appropriate masks to take care of envs that have set dones to True & learn() accordingly
        episode = 0
        cum_step_rewards = np.zeros(self.params["num_agents"])
        episode_rewards = []
        step_num = 0
        while True:
            action = self.get_action(obs)
            next_obs, rewards, dones, _ = self.envs.step(action)
            self.rewards.append(torch.tensor(rewards))
            done_env_idxs = np.where(dones)[0]
            cum_step_rewards += rewards  # nd-array of shape=num_actors

            step_num += self.params["num_agents"]
            episode += done_env_idxs.size  # Update the number of finished episodes
            if not args.test and(step_num >= self.params["learning_step_thresh"] or done_env_idxs.size):
                self.learn(next_obs, dones)
                step_num = 0
                # Monitor performance and save Agent's state when perf improves
                if done_env_idxs.size > 0:
                    [episode_rewards.append(r) for r in cum_step_rewards[done_env_idxs] ]
                    if np.max(cum_step_rewards[done_env_idxs]) > self.best_reward:
                        self.best_reward = np.max(cum_step_rewards[done_env_idxs])
                    if np.mean(episode_rewards) > prev_checkpoint_mean_ep_rew:
                        num_improved_episodes_before_checkpoint += 1
                    if num_improved_episodes_before_checkpoint >= self.params["save_freq_when_perf_improves"]:
                        prev_checkpoint_mean_ep_rew = np.mean(episode_rewards)
                        self.best_mean_reward = np.mean(episode_rewards)
                        self.save()
                        num_improved_episodes_before_checkpoint = 0

                    writer.add_scalar(self.actor_name + "/mean_ep_rew", np.mean(cum_step_rewards[done_env_idxs]),
                                      self.global_step_num)
                    # Reset the cum_step_rew for the done envs
                    cum_step_rewards[done_env_idxs] = 0.0

            obs = next_obs
            self.global_step_num += self.params["num_agents"]
            if args.render:
                self.envs.render()
            #print(self.actor_name + ":Episode#:", episode, "step#:", step_num, "\t rew=", reward, end="\r")
            writer.add_scalar(self.actor_name + "/reward", np.mean(cum_step_rewards), self.global_step_num)
            print("{}:Episode#:{} \t avg_step_reward:{:.4} \t mean_ep_rew:{:.4}\t best_ep_reward:{:.4}".format(
                self.actor_name, episode, np.mean(cum_step_rewards), np.mean(episode_rewards), self.best_reward))


if __name__ == "__main__":
    agent_params = params_manager.get_agent_params()
    agent_params["model_dir"] = args.model_dir
    mp.set_start_method('spawn')

    env_names = [args.env] * agent_params["num_agents"]

    agent = DeepActorCriticAgent(0, env_names , agent_params)
    agent.run()
