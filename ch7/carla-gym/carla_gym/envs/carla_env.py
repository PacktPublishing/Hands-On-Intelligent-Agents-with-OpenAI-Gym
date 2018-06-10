"""
OpenAI Gym compatible Driving simulation environment based on Carla.
Requires the system environment variable CARLA_SERVER to be defined and be pointing to the
CarlaUE4.sh file on your system. The default path is assumed to be at: ~/software/CARLA_0.8.2/CarlaUE4.sh
"""

import atexit
import os
import random
import signal
import subprocess
import time
import traceback
import json
import numpy as np
import gym
from gym.spaces import Box, Discrete, Tuple

# Set this to the path to your Carla binary
SERVER_BINARY = os.environ.get(
    "CARLA_SERVER", os.path.expanduser("~/software/CARLA_0.8.2/CarlaUE4.sh"))
assert os.path.exists(SERVER_BINARY), "CARLA_SERVER environment variable is not set properly. Please check and retry"
# Import Carla python client API funcs
from .carla.client import CarlaClient
from .carla.sensor import Camera
from .carla.settings import CarlaSettings
from .carla.planner.planner import Planner, REACH_GOAL, GO_STRAIGHT, \
    TURN_RIGHT, TURN_LEFT, LANE_FOLLOW

# Carla planner commands
COMMANDS_ENUM = {
    REACH_GOAL: "REACH_GOAL",
    GO_STRAIGHT: "GO_STRAIGHT",
    TURN_RIGHT: "TURN_RIGHT",
    TURN_LEFT: "TURN_LEFT",
    LANE_FOLLOW: "LANE_FOLLOW",
}

# Mapping from string repr to one-hot encoding index to feed to the model
COMMAND_ORDINAL = {
    "REACH_GOAL": 0,
    "GO_STRAIGHT": 1,
    "TURN_RIGHT": 2,
    "TURN_LEFT": 3,
    "LANE_FOLLOW": 4,
}
