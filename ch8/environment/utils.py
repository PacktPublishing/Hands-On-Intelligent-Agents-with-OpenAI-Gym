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



