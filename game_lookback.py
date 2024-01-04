import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from direction import Direction
from laser import Laser
from levels import Levels
from agent import Agent
from game import Game

class Game2D(Game):

    def __init__(self, config):
        super().__init__(config)
        self.name = "2D"
        lookback = config.get("lookback", 64)
        self.observation = np.zeros((82, 1, lookback))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(82, 1, lookback))

    def _update_obs(self):
        obs = self._get_obs()
        self.observation[:, :, :-1] = self.observation[:, :, 1:]
        self.observation[:, :, -1] = obs

