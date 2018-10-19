### Custom learning environment implementation that is compatible with the OpenAI Gym interface

*UPDATE:* Checkout the [CARLA Gym environment space description](#2-carla-gym) below that you may find useful
to setup the CARLA learning environment for RL agent development

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
9 different discrete actions which along with other parameters are configurable using the `ENV_CONFIG` file. The script
 also prints out the environment step number, instantaneuous reward, cumulative episode rewards and the value of done (True/False 
signifying the end of an episode).  

##### 2.1 State space and Action space definitions

- ###### Discrete Actions

If `"discrete_actions": True` is set in the `ENV_CONFIG`, the following space definitions
are used by the CARLA Gym environment:

|Item | Space| Description |
|--------|----------|------------------|
|`state_space`| `Box(low=0, high=255, shape=(WIDTH, HEIGHT, CHANNELS), dtype=np.uint8)` | Image from Camera(s). RGB is preferred. Frames from multiple RGB/Depth cameras can be stacked along the CHANNEL dimension|
| `action_space`| `Discrete(9)` | Discrete driving actions (0 to 9). See below for detailed description |

The discrete actions have the following meaning:

    0: [0.0, 0.0],    # Coast
    1: [0.0, -0.5],   # Turn Left
    2: [0.0, 0.5],    # Turn Right
    3: [1.0, 0.0],    # Forward
    4: [-0.5, 0.0],   # Brake
    5: [1.0, -0.5],   # Bear Left & accelerate
    6: [1.0, 0.5],    # Bear Right & accelerate
    7: [-0.5, -0.5],  # Bear Left & decelerate
    8: [-0.5, 0.5],   # Bear Right & decelerate
The float values in the table above represent [steer, throttle + brake] values 
that map to the continuous action space (see below).

- ###### Continuous Actions

If `"discrete_actions": False` is set in the `ENV_CONFIG` then, the following space definitions are
used by the CARLA Gym environment:


|Item | Space| Description |
|--------|----------|------------------|
|`state_space`| `Box(low=0, high=255, shape=(WIDTH, HEIGHT, CHANNELS), dtype=np.uint8)` | Image from Camera(s). RGB is preferred. Frames from multiple RGB/Depth cameras can be stacked along the CHANNEL dimension|
| `action_space`| `Box(low=-1, high=1, shape=(2,), dtype=np.float32)` | steering: [-1, 1], throttle: [0, 1], brake: [-1, 0]. Throttle & brake together share the second dimension |


The scenarios of the driving scene is also customizable and is defined in `carla-gym/carla_gym/envs/scenarios.json` file.

Take a look at [Chapter 8](https://github.com/PacktPublishing/Hands-On-Intelligent-Agents-with-OpenAI-Gym/tree/master/ch8)
for training reinforcement learning agents in the CARLA Gym environment.
