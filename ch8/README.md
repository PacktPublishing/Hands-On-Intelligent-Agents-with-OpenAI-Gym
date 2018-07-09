### Deep Actor-Critic Agent
Contains an easy-to-follow, simple-to-use Actor-Critic architecture based reinforcement learning agent implementations that uses a deep neural
network to represent the policy and value function. Following are the agent algorithms discussed and implemented in this chapter:
  - Deep n-step Advantage Actor Critic Agent
  - Asynchronous Deep n-step Advantage Actor Critic Agent
  - Batched Deep n-step Advantaged Actor Critic Agent

### Code Structure
[To be updated]

### Pre-trained Agent brains
Trained agents brains/models and instructions to test them can be found in the [trained_models](trained_models) directory.

#### (optional) Requirements:
If you use the`CarRacing` environment to train or test the agent, you will need the following additional 
python packages:
  - Python 3.5 (because Box2D from the author's official anaconda channel (kne) has binaries built only for py upto 3.5)
  - Box2D. Can be installed using: 
  `conda install -c kne pybox2d`