#!/usr/bin/env python
from collections import namedtuple
import random

Experience = namedtuple("Experience", ['obs', 'action', 'reward', 'next_obs',
                                       'done'])


class ExperienceMemory(object):
    """
    """
    def __init__(self, capacity=int(1e6)):
        """

        :param capacity: Total capacity (Max number of Experiences)
        :return:
        """
        self.capacity = capacity
        self.mem_idx = 0  # Index of the current experience
        self.memory = []

    def store(self, experience):
        """

        :param experience: The Experience object to be stored into the memory
        :return:
        """
        self.memory.insert(self.mem_idx % self.capacity, experience)
        self.mem_idx += 1

    def sample(self, batch_size):
        """

        :param batch_size:  Sample batch_size
        :return: A list of batch_size number of Experiences sampled at random from mem
        """
        assert batch_size <= len(self.memory), "Sample batch_size is more than available exp in mem"
        return random.sample(self.memory, batch_size)

    def get_size(self):
        """

        :return: Number of Experiences stored in the memory
        """
        return len(self.memory)
