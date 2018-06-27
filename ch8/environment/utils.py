#!/usr/bin/env python
import multiprocessing as mp
import gym

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
