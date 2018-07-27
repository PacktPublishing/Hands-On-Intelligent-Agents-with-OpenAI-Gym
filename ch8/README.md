# Chapter 8: Implementing an Intelligent & Autonomous Car Driving Agent using Deep Actor-Critic Algorithm

1. [Brief chapter summary and outline of topics covered]()
2. [Code Structure](#2-code-structure)
3. [Running the code](#3-running-the-code)
   * [Deep n-step Advantage Actor-Critic](#deep-n-step-advantage-actor-critic-agent)
       - [Training](#a2c-training)
       - [Testing](#a2c-testing)
   * [Asynchronous Deep n-step Advantage Actor-Critic](#asynchronous-deep-n-step-advantage-actor-critic-agent)
       - [Training](#async-a2c-training)
       - [Testing](#async-a2c-testing)

## 2. Code structure

 * [a2c_agent.py](./a2c_agent.py) --> Main script to launch the deep n-step Advantage Actor-Critic (A2C) agent
 * [a2c_parameters.json](./a2c_parameters.json)  --> Configuration parameters for the [a2c_agent](./a2c_agent.py) and the environment
 * [async_a2c_agent.py](./async_a2c_agent.py)  --> Main script to launch the deep n-step Asynchronous Advantage Actor-Critic (A3C) agent
 * [async_a2c_parameters.json](./async_a2c_parameters.json)  --> Configuration parameters for the [async_a2c_agent.py](./async_a2c_agent.py) and the environment
 * [batched_a2c_agent.py](./batched_a2c_agent.py)  --> Example script showing how agents can be run in parallel with batches of environments
 * [environment](./environment)  --> Module containing environment implementations, wrapper and interfaces
   * [atari.py](./environment/atari.py)  --> Wrappers and env pre-processing functions for the Atari Gym environment
   * [carla_gym](./environment/carla_gym)  --> OpenAI Gym compatible Carla driving environment module (see [chapter 7](../ch7/) for more details about impl)
     * [envs](./environment/carla_gym/envs)  --> the Carla Gym environment
       * [carla](./environment/carla_gym/envs/carla)  --> Refer to [Chapter 7](../ch7) for implementation details
       * [carla_env.py](./environment/carla_gym/envs/carla_env.py)  --> Carla driving environment implementation
       * [__init__.py](./environment/carla_gym/envs/__init__.py)
       * [scenarios.json](./environment/carla_gym/envs/scenarios.json)  --> Carla environment configuration parameters to change the driving scenarios: map/city, weather conditions, route etc.
     * [\_\_init__.py](./environment/carla_gym/__init__.py)
   * [\_\_init__.py](./environment/__init__.py)
   * [utils.py](./environment/utils.py)  --> Utilities to vectorize and run environment instances in parallel as separate processes
 * [function_approximator](./function_approximator)  --> Module with neural network implementations
   * [deep.py](./function_approximator/deep.py)  --> Deep neural network implementations in PyTorch for policy and value function approximation
   * [\_\_init__.py](./function_approximator/__init__.py)
   * [shallow.py](./function_approximator/shallow.py)  --> Shallow neural network implementations in PyTorch for policy and value function approximations
 * [logs](./logs)  --> Folder to contain the Tensorboard log files for each run (or experiment)
   * ENVIRONMENT_NAME_RUN_TIMESTAMP*  --> Folder created for each run based on environment name and the run timestamp
     * `agent_params.json`  --> The parameters used by the agent corresponding to this run/experiment
     * `env_params.json`  --> The environment configuration parameters used in this run/experiment
     * `events.out.tfevents.*`  --> Tensorboard event log files
