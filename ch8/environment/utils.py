#!/usr/bin/env python
import multiprocessing as mp
import gym
from abc import ABC, abstractmethod
import numpy as np

class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    """
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a tuple of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    @abstractmethod
    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def render(self):
        print('WARNING:Render not defined for %s'%self)

    @property
    def unwrapped(self):
        return self


def make_env_in_sep_proc(env_name, shared_pipe, parent_pipe, stack=False, scale_rew=False):
    """
    Create an environment instance (remote or local) in a separate proc and return the env object
    :return: The env running in a different proc
    """
    parent_pipe.close()

    env = gym.make(env_name)
    # Apply env pre-processing here if needed
    #if scale_rew:
    #    env = RewardScaler(env)
    #env = CustomWarpFrame(env)
    #env = NormalizedEnv(env)

    while True:
        method, data = shared_pipe.recv()
        if method == 'step':
            next_obs, rew, done, info = env.step(data)
            if done:
                next_obs = env.reset()
            shared_pipe.send((next_obs, rew, done, info))

        if method == 'reset':
            obs = env.reset()
            shared_pipe.send(obs)

        if method == 'get_spaces':
            shared_pipe.send((env.observation_space, env.action_space))



class EnvProc(mp.Process):
    def __init__(self, env_name, requests):
        super(EnvProc, self).__init__()
        self.env_name = env_name
        self.requests= requests
        self.terminate = False

    def run(self):
        self.env = gym.make(self.env_name)
        while not self.terminate:
            #while not self.request_queue.empty() or self.request_queue.qsize():
            request = self.requests.recv()
            result = self.call_env(request["method"], request["data"])
            self.requests.send(result)

    def call_env(self, method, data):
        if method == "step":
            next_obs, reward, done, info = self.env.step(data)
            return (next_obs, reward, done, info)
        elif method == "reset":
            obs = self.env.reset()
            return obs
        elif method == "render":
            self.env.render()
        elif method == "observation_space":
            return self.env.observation_space
        elif method == "action_space":
            return self.env.action_space
        elif method == "close":
            self.env.close()
            self.terminate = True


class EnvProxy(object):
    def __init__(self, env_name):
        self.pipe, self.child_pipe = mp.Pipe()
        self.env_proc = EnvProc(env_name, self.child_pipe)
        self.env_proc.start()
    def step(self, action):
        self.pipe.send({"method": "step", "data": action})
        return self.pipe.recv()
    def reset(self):
        self.pipe.send({"method": "reset", "data": None})
        return self.pipe.recv()
    def render(self):
        self.pipe.send({"method": "render", "data": None})
    @property
    def observation_space(self):
        self.pipe.send({"method": "observation_space", "data": None})
        return self.pipe.recv()
    @property
    def action_space(self):
        self.pipe.send({"method": "action_space", "data": None})
        return self.pipe.recv()
    def close(self):
        self.pipe.send({"method": "close", "data": None})
        self.env_proc.join()


def make_env(env_name):
    env_proxy= EnvProxy(env_name)
    return env_proxy
