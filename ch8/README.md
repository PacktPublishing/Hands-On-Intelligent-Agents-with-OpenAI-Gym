### Deep Actor-Critic Agent
[To be updated]

Contains an easy-to-follow, simple-to-use policy gradient based reinforcement learning agent implementations that uses a deep neural
network to represent the policy and value function and learns using temporal difference learning algorithm

### Pre-trained Agent brains
Trained agents brains/models and instructions to test them can be found in the [trained_models](trained_models) directory.

#### (optional) Requirements:
If you use the`CarRacing` environment to train or test the agent, you will need the following additional 
python packages than what is listed in the [conda_env.yaml](../conda_env.yaml) requirements:
  - Python 3.5 (because Box2D from the author's official anaconda channel (kne) has binaries built only for py upto 3.5)
  - Box2D. Can be installed using: 
  `conda install -c kne pybox2d`