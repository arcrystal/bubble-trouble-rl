import gym
from gym.spaces import Discrete, Dict, Box
import numpy as np
import os

DISPLAY_WIDTH = float(os.environ.get('DISPLAY_WIDTH'))
DISPLAY_HEIGHT = DISPLAY_WIDTH * 0.5337 # Default 475

class Environment(gym.Env):
    def __init__(self, env_config={}):
        self.observation_space = Dict({
            "posX": Discrete(int(DISPLAY_WIDTH)),
            "velX": Discrete(3),
            "balls": Box(
                low=np.array([0., 0.]),
                high=np.array([int(DISPLAY_WIDTH), int(DISPLAY_HEIGHT)]),
                dtype=np.uint8)
        })
        # Shoot, left, right, do nothing
        self.action_space = gym.spaces.Discrete(4)

    def reset(self):
        pass

    def step(self, action):
        pass

env = Environment()
b = env.observation_space.sample()
print(b)