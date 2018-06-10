### Custom learning environment implementation that is compatible with the OpenAI Gym interface

There are currently two custom OpenAI Gym compatible environment implementation in this folder:

  1.`custom-environment`
  
  2.`carla_gym`
  

#### 1. custom-environment

This is a template for building a custom OpenAI Gym compatible environments which also includes the necessary init hooks
in place to register the environment with the Gym registry so that we can use the usual `gym.make(ENV_NAME)` to create
an instance of the custom environment.

As an example, you can do the following from the current directory in a bash shell:

`$cd custom-environments`

`$python`

```python
>>> import gym
>>> import custom_environments
>>> env = gym.make("CustomEnv-v0")
>>> obs = env.reset()
```

And the remaining flow with the Gym environment can follow as usual.


#### 2. carla-gym

`carla-gym` folder contains a custom implementation of an OpenAI Gym compatible driving environment based on Carla.
The environment implementation is in the `carla-gym/carla_gym/envs/carla_env.py` file. Which can be directly run as 
a script to test the functionality like below:

`python carla-gym/carla_gym/envs/carla_env.py` which should launch a Carla driving simulator instance and run for 5 episodes
using a naive (hard-coded) agent with a constant action of going forward with full throttle. The environment can be 
configured to have a continuous action space with values for steering, throttle and brake or a discrete action space with 
8 different discrete actions which along with other parameters are configurable using the `ENV_CONFIG` file. The script
 also prints out the environment step number, instantaneuous reward, cumulative episode rewards and the value of done (True/False 
signifying the end of an episode).  

(TODO: Add screenshots)

The scenarios of the driving scene is also customizable and is defined in `carla-gym/carla_gym/envs/scenarios.json` file.
