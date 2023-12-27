import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from direction import Direction
from laser import Laser
from levels import Levels
from agent import Agent
from game import Game


class Game2DFlat(Game):

    def __init__(self, config):
        super().__init__(config)
        self.name = "2DFlat"
        lookback = config.get("lookback", 64)
        self.observation_box = np.zeros((82, lookback))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(82*lookback,))
        self.observation_box = np.zeros((82, lookback))
        self.observation = np.zeros(self.observation_space.shape)


    def _update_obs(self):
        obs = self._get_obs()
        self.observation_box[:, :-1] = self.observation_box[:, 1:]
        self.observation_box[:, -1] = obs
        self.observation = self.observation_box.reshape(self.observation_space.shape)
