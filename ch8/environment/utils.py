#!/usr/bin/env python
# Vectorized environment implementation based on OpenAI Gym| Praveen Palanisamy
# Chapter 8, Hands-on Intelligent Agents with OpenAI Gym, 2018

import multiprocessing as mp
import gym
from abc import ABC, abstractmethod
import numpy as np
import cv2

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


class ResizeFrame(gym.ObservationWrapper):
    def __init__(self,env):
        gym.ObservationWrapper.__init__(self, env)
        self.desired_width = 84  # Change this as necessary. 84 is not a magic number.
        self.desired_height = 84
    def observation(self, obs):
        if len(obs.shape) == 3:  # Observations are image frames
            obs = cv2.resize(obs, (self.desired_width, self.desired_height))
        return obs

def run_env_in_sep_proc(env_name, shared_pipe, parent_pipe, stack=False, scale_rew=False):
    """
    Create and run an environment instance (remote or local) in a separate proc
    """
    parent_pipe.close()

    env = gym.make(env_name)
    # Apply env pre-processing here if needed
    #if scale_rew:
    #    env = RewardScaler(env)
    #env = CustomWarpFrame(env)
    #env = NormalizedEnv(env)
    env = ResizeFrame(env)

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

class SubprocVecEnv(VecEnv):
    def __init__(self, env_names, spaces=None):
        """
        env_names: list of (gym) environments to run in sub/separate processes
        """
        self.waiting = False
        self.closed = False
        num_envs = len(env_names)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.ps = []
        for (env_name, worker_conn, parent_conn) in zip(env_names, self.work_remotes, self.remotes):
            self.ps.append(mp.Process(target=run_env_in_sep_proc, args=(env_name, worker_conn, parent_conn)))
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()

        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, num_envs, observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


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
