# Chapter 8: Implementing an Intelligent & Autonomous Car Driving Agent using Deep Actor-Critic Algorithm

1. [Brief chapter summary and outline of topics covered](1-brief-chapter-summary-and-outline-of-topics-covered)
2. [Code Structure](#2-code-structure)
3. [Running the code](#3-running-the-code)
   * [Deep n-step Advantage Actor-Critic](#deep-n-step-advantage-actor-critic-agent)
       - [Training](#a2c-training)
       - [Testing](#a2c-testing)
   * [Asynchronous Deep n-step Advantage Actor-Critic](#asynchronous-deep-n-step-advantage-actor-critic-agent)
       - [Training](#async-a2c-training)
       - [Testing](#async-a2c-testing)

[![HOIAWOG A3C Carla 9](https://praveenp.com/projects/HOIAWOG/ch8_training_9async_carla.gif)](https://praveenp.com/blog/#hands-on-intelligent-agents-with-openai-gym-hoiawog)

A sample screencapture showing 9 agents training asynchronously launched using the [async_a2c.py](async_a2c.py) script with `num_agents` parameter in [async_a2c_parameters.json](async_a2c_parameters.json) set to `9`. (Refer to [Async A2C Training section](#async-a2c-training) for the command used to launch the training)


## 1. Brief chapter summary and outline of topics covered
This chapter teaches you the fundamentals of the Policy Gradient based reinforcement learning algorithms and helps you intuitively understand the deep n-step advantage actor-critic algorithm. You will then learn to implement a super-intelligent agent that can drive a car autonomously in the Carla simulator using both the synchronous as well as asynchronous implementation of the deep n-step advantage actor-critic algorithm.

Following is a higher-level outline of the topics covered in this chapter:

* Deep n-step Advantage Actor-Critic algorithm
  * Policy Gradients
    * The likelyhood ratio trick
    * The policy gradient theorem
  * Actor-Critic algorithms
  * Advantage Actor-Critic algorithm
  * n-step Advantage Actor-Critic algorithm
    * n-step returns
    * Implementing the n-step return calculation
* Implementing deep n-step Advantage Actor-Critic algorithm
* Training an intelligent and autonomous driving agent
  * Training the agent to drive a car in the CARLA driving simulator

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
 * [README.md](./README.md)
 * [trained_models](./trained_models)  --> Folder containing trained-models/"brains" for the agents
   * [README.md](./trained_models/README.md)  --> Description of the trained agent "brains"/models with the naming conventions
 * [utils](./utils)  --> Module containing utility functions to train/test the agent
     * [params_manager.py](./utils/params_manager.py)  --> A simple class to manage the agent's and environment's parameters

## 3. Running the code

* ### Deep n-step Advantage Actor-Critic Agent:

    The [a2c_agent.py](./a2c_agent.py) is the main script that takes care of the training and testing of the deep n-step advantage Actor-Critic agent.
    The table below summarizes the argument that the script supports and what they mean. Note that, most of the agent and environment
    related configuration parameters are in the [a2c_parameters.json](./a2c_parameters.json) file and only those few parameters that are more useful
    when launching the training/testing scripts are made available through the command line interface.

    | Argument      | Description |
    |---------------| ------------------------ |
    |`--env`        | Name of the OpenAI Gym interface compatible environment. Use `Carla-v0` if you want to train/test in the Carla driving environment. Supports other Gym environments as well|
    |`--params-file`| Path to the json parameters file. Default=`./a2c_parameters.json`|   
    |`--model-dir`  | Directory to save/load trained agent brain/model. Default=`./trained_models`|
    |`--render`     | True/False to render/not-render the environment to the display|
    |`--test`       | Run the agent in test mode using a saved trained brain/model. Learning is disabled|
    |`--gpu-id`     | GPU device ID to use. Default=0|
    
    #### A2C Training
    Make sure the `rl_gym_book` conda environment with the necessary packages installed is activated. Assuming that you cloned 
    the code as per the instructions to `~/HOIAWOG/`,  you can launch the Agent training script from the `~/HOIAWOG/ch8` directory using the following command:
    
    `python a2c_agent.py --env Carla-v0 --gpu-id 0` 
    
    If a saved agent "brain" (trained model) is available for the chosen environment, the training script will upload 
    that brain to the agent and continue training the agent to improve further.
    
    The log files are written to the directory pointed with the `summary_file_path_prefix` parameter (the default is `logs/A2C_`). When the training script is running, you can monitor the learning progress of the agent visually using Tensorboard. From the `~/HOIAWOG/ch6` directory, you can launch Tensorboard with the following command: `tensorboard --log_dir=./logs/`. 
    You can then visit the web URL printed on the console (the default one is: http://localhost:6006) to monitor the progress.
    
    You can train the agent in *any* Gym compatible environment by providing the Gym env ID for the `--env` argument.
    Listed below are some short list of environments that you can train the agent in:
    
    |Environment Types                                                   | Example command to train                                  |
    |--------------------------------------------------------------------| ----------------------------------------------------------|
    | [Gym classic control](https://gym.openai.com/envs/#classic_control)| `python a2c_agent.py --env Acrobot-v1 --gpu-id 0`        |
    | [Gym Box2D](https://gym.openai.com/envs/#box2d)                    |`python a2c_agent.py --env LunarLander-v2 --gpu-id 0`    |
    | [Gym Atari environment](https://gym.openai.com/envs/#atari)        | `python a2c_agent.py --env AlienNoFrameskip-v4 --gpu-id 0`|
    | [Roboschool](https://github.com/openai/roboschool)                 | `python a2c_agent.py --env RoboschoolHopper-v1 --gpu-id 1`| 
    
    #### A2C Testing
    Make sure the `rl_gym_book` conda environment with the necessary packages installed is activated. Assuming that you cloned 
    the code as per the instructions to `~/HOIAWOG/`,  you can launch the Agent testing script from the `~/HOIAWOG/ch8` directory using the following command:
    
    `python a2c_agent.py --env Carla-v0 --test --render`
    
    The above command will launch the agent in testing mode by uploading the saved brain state (if available) for this environment
    to the agent. The `--test` argument disables learning and simply evaluates the agent's performance in the chosen environment.
    
    You can test the agent in any OpenAI Gym interface compatible learning environment like with the training procedure.
    Listed below are some example environments from the list of environments for which trained brains/models are made available
    in this repository:
    
    |Environment Types                                                   | Example command to train                                  |
    |--------------------------------------------------------------------| ----------------------------------------------------------|
    | [Gym classic control](https://gym.openai.com/envs/#classic_control)| `python a2c_agent.py --env Pendulum-v0 --test --render`        |
    | [Gym Box2D](https://gym.openai.com/envs/#box2d)                    |`python a2c_agent.py --env BipedalWalker-v2 --test --render`    |
    | [Gym Atari environment](https://gym.openai.com/envs/#atari)        | `python a2c_agent.py --env RiverraidNoFrameskip-v4 --test --render --gpu-id 0`|
    | [Roboschool](https://github.com/openai/roboschool)                 | `python a2c_agent.py --env RoboschoolHopper-v1 --test --render --gpu-id 1`| 
    

* ### Asynchronous Deep n-step Advantage Actor-Critic Agent:

    The [async_a2c_agent.py](./async_a2c_agent.py) is the main script that takes care of the training and testing of the asynchronous deep n-step advantage Actor-Critic agent.
    The table below summarizes the argument that the script supports and what they mean. Note that, most of the agent and environment
    related configuration parameters are in the [async_a2c_parameters.json](./async_a2c_parameters.json) file and only those few parameters that are more useful
    when launching the training/testing scripts are made available through the command line interface.
    
    | Argument      | Description |
    |---------------| ------------------------ |
    |`--env`        | Name of the Gym-compatible environment. Use `Carla-v0` if you want to train/test in the Carla driving environment. Supports other Gym environments as well|
    |`--params-file`| Path to the json parameters file. Default=`./async_a2c_parameters.json`|   
    |`--model-dir`  | Directory to save/load trained agent brain/model. Default=`./trained_models`|
    |`--render`     | True/False to render/not-render the environment to the display|
    |`--test`       | Run the agent in test mode using a saved trained brain/model. Learning is disabled|
    |`--gpu-id`     | GPU device ID to use. Default=0|

    #### Async A2C Training
    
    **NOTE:** Because this agent training script will spawn multiple agents and environment instances, make sure
    you set the `num_agents` parameter in [async_a2c_parameters.json](./async_a2c_parameters.json) file to sensible values based
    on the hardware of the machine on which you are running this script. If you are using the `Carla-v0` environment to
    train the agent in the Carla driving environment, be aware that the Carla server instance itself needs some GPU resource to run 
    on top of the agent's resource needs.
    
    Make sure the `rl_gym_book` conda environment with the necessary packages installed is activated. Assuming that you cloned 
    the code as per the instructions to `~/HOIAWOG/`,  you can launch the Agent training script from the `~/HOIAWOG/ch8` directory using the following command:
    
    `python async_a2c_agent.py --env Carla-v0 --gpu-id 0` 
    
    The screencapture animation (GIF) at the top of this page was captured by launching the above command with `num_agents` in [async_a2c_parameters.json](./async_a2c_parameters.json) set to `9`.
    
    If a saved agent "brain" (trained model) is available for the chosen environment, the training script will upload 
    that brain to the agent and continue training the agent to improve further.
    
    The log files are written to the directory pointed with the `summary_file_path_prefix` parameter (the default is `logs/A2C_`). When the training script is running, you can monitor the learning progress of the agent visually using Tensorboard. From the `~/HOIAWOG/ch6` directory, you can launch Tensorboard with the following command: `tensorboard --log_dir=./logs/`. 
    You can then visit the web URL printed on the console (the default one is: http://localhost:6006) to monitor the progress.
    
    You can train the agent in *any* Gym compatible environment by providing the Gym env ID for the `--env` argument.
    Listed below are some short list of environments that you can train the agent in:
    
    |Environment Types                                                   | Example command to train                                  |
    |--------------------------------------------------------------------| ----------------------------------------------------------|
    | [Gym classic control](https://gym.openai.com/envs/#classic_control)| `python async_a2c_agent.py --env Acrobot-v1 --gpu-id 0`        |
    | [Gym Box2D](https://gym.openai.com/envs/#box2d)                    |`python async_a2c_agent.py --env LunarLander-v2 --gpu-id 0`    |
    | [Gym Atari environment](https://gym.openai.com/envs/#atari)        | `python async_a2c_agent.py --env SeaquestNoFrameskip-v4 --gpu-id 0`|
    | [Roboschool](https://github.com/openai/roboschool)                 | `python async_a2c_agent.py --env RoboschoolHopper-v1 --gpu-id 1`| 
    
    #### Async A2C Testing
    Make sure the `rl_gym_book` conda environment with the necessary packages installed is activated. Assuming that you cloned 
    the code as per the instructions to `~/HOIAWOG/`,  you can launch the Agent testing script from the `~/HOIAWOG/ch8` directory using the following command:
    
    `python async_a2c_agent.py --env Carla-v0 --test`
    
    The above command will launch the agent in testing mode by uploading the saved brain state (if available) for this environment
    to the agent. The `--test` argument disables learning and simply evaluates the agent's performance in the chosen environment.
    
    You can test the agent in any OpenAI Gym interface compatible learning environment like with the training procedure.
    Listed below are some example environments from the list of environments for which trained brains/models are made available
    in this repository:
    
    |Environment Types                                                   | Example command to train                                  |
    |--------------------------------------------------------------------| ----------------------------------------------------------|
    | [Gym classic control](https://gym.openai.com/envs/#classic_control)| `python async_a2c_agent.py --env Pendulum-v0 --test --render`        |
    | [Gym Box2D](https://gym.openai.com/envs/#box2d)                    |`python async_a2c_agent.py --env LunarLander-v2 --test --render`    |
    | [Gym Atari environment](https://gym.openai.com/envs/#atari)        | `python async_a2c_agent.py --env RiverraidNoFrameskip-v4 --test --render --gpu-id 0`|
    | [Roboschool](https://github.com/openai/roboschool)                 | `python async_a2c_agent.py --env RoboschoolHopper-v1 --test --render --gpu-id 1`| 
