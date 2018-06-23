import torch


class Actor(torch.nn.Module):
    def __init__(self, input_shape, output_shape, device=torch.device("cpu")):
        """
        A feed forward neural network that produces two continuous values mean (mu) and sigma, each of output_shape
        . Used to represent the Actor in an Actor-Critic algorithm
        :param input_shape: Shape of the inputs. This is typically the shape of each of the observations for the Actor
        :param output_shape: Shape of the outputs. This is the shape of the actions that the Actor should produce
        :param device: The torch.device (cpu or cuda) where the inputs and the parameters are to be stored and operated
        """
        super(Actor, self).__init__()
        self.device = device
        self.layer1 = torch.nn.Sequential(torch.nn.Linear(input_shape[0], 64),
                                          torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Linear(64, 32),
                                          torch.nn.ReLU())
        self.actor_mu = torch.nn.Linear(32, output_shape)
        self.actor_sigma = torch.nn.Linear(32, output_shape)

    def forward(self, x):
        """
        Forward pass through the Actor network. Takes batch_size x observations as input and produces mu and sigma
        as the outputs
        :param x: The observations
        :return: Mean (mu) and Sigma (sigma) for a Gaussian policy
        """
        x = x.to(self.device)
        x = self.layer1(x)
        x = self.layer2(x)
        mu = self.actor_mu(x)
        sigma = self.actor_sigma(x)
        return mu, sigma


class Critic(torch.nn.Module):
    def __init__(self, input_shape, output_shape=1, device=torch.device("cpu")):
        """
        A feed forward neural network that produces a continuous value. Used to represent the Critic
        in an Actor-Critic algorithm that estimates the value of the current observation/state
        :param input_shape: Shape of the inputs. This is typically the shape of the observations for the Actor
        :param output_shape: Shape of the output. This is most often 1 as the Critic is expected to produce a single
        value given given an observation/state
        :param device: The torch.device (cpu or cuda) where the inputs and the parameters are to be stored and operated
        """
        super(Critic, self).__init__()
        self.device = device
        self.layer1 = torch.nn.Sequential(torch.nn.Linear(input_shape[0], 64),
                                          torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Linear(64, 32),
                                          torch.nn.ReLU())
        self.critic= torch.nn.Linear(32, output_shape)

    def forward(self, x):
        """
        Forward pass through the Critic network. Takes batch_size x observations as input and produces the value
        estimate as the output
        :param x: The observations
        :return: Mean (mu) and Sigma (sigma) for a Gaussian policy
        """
        x = x.to(self.device)
        x = self.layer1(x)
        x = self.layer2(x)
        critic = self.critic(x)
        return critic


class ActorCritic(torch.nn.Module):
    def __init__(self, input_shape, actor_shape, critic_shape, device=torch.device("cpu")):
        """
        A feed forward neural network used to represent both an Actor and the Critic in an Actor-Critic algorithm.
        :param input_shape: Shape of the inputs. This is typically the shape of the observations
        :param actor_shape: Shape of the actor outputs. This is the shape of the actions that the Actor should produce
        :param critic_shape: Shape of the critic output. This is most often 1 as the Critic is expected to produce a
        single value given given an observation/state
        :param device: The torch.device (cpu or cuda) where the inputs and the parameters are to be stored and operated
        """
        super(ActorCritic, self).__init__()
        self.device = device
        self.layer1 = torch.nn.Sequential(torch.nn.Linear(input_shape[0], 32),
                                          torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.Linear(32, 16),
                                          torch.nn.ReLU())
        self.actor_mu = torch.nn.Linear(16, actor_shape)
        self.actor_sigma = torch.nn.Linear(16, actor_shape)
        self.critic = torch.nn.Linear(16, critic_shape)

    def forward(self, x):
        """
        Forward pass through the Actor-Critic network. Takes batch_size x observations as input and produces
        mu, sigma and the value estimate
        as the outputs
        :param x: The observations
        :return: Mean (actor_mu), Sigma (actor_sigma) for a Gaussian policy and the Critic's value estimate (critic)
        """
        x.requires_grad_()
        x = x.to(self.device)
        x = self.layer1(x)
        x = self.layer2(x)
        actor_mu = self.actor_mu(x)
        actor_sigma = self.actor_sigma(x)
        critic = self.critic(x)
        return actor_mu, actor_sigma, critic

