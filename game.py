import gymnasium as gym
import numpy as np
import pygame
import math

from direction import Direction
from laser import Laser
from levels import Levels
from player import Player

class Game(gym.Env):

    def __init__(self, render_mode=None):
        self.steps = 0
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.width = 640
        self.height = 480
        self.reward_range = (-math.inf, 3+7+15+(7+7)+(7+15)+(6*1)+(7*1)+31)

        if render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()
        else:
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.HIDDEN)

        self.fps = 48
        self.timestep = 1 / self.fps
        self.emptyScreen = np.zeros((self.height, self.width, 3))
        self.levels = Levels(self.width, self.height, self.fps)
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(self.height, self.width, 3),
                                                dtype=np.uint8)
        self.reset()
        self._action_to_direction = {
            0: Direction.LEFT,
            1: Direction.RIGHT,
            2: Direction.SHOOT,
            3: Direction.STILL,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        observation = self.emptyScreen
        info = {}
        self.player = Player(self.width / 2, self.height, self.width)
        self.player.laser = Laser(self.width, self.height)
        self.level = 1
        self.balls = self.levels.get(self.level)
        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action=None):
        # self.steps += 1
        # print("Steps :", self.steps)
        observation = None
        reward = -0.05
        terminated = False
        truncated = False
        info = {}

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        if action is None:
            keys = pygame.key.get_pressed()
            self.player.step(Direction.STILL)
            if keys[pygame.K_LEFT]:
                self.player.step(Direction.LEFT)
            if keys[pygame.K_RIGHT]:
                self.player.step(Direction.RIGHT)
            if keys[pygame.K_UP]:
                self.player.step(Direction.SHOOT)
        else:
            direction = self._action_to_direction[action]
            self.player.step(direction)

        self.player.laser.update()
        self.balls.update()
        self.screen.fill((0, 0, 0))
        self.screen.blit(self.player.image, self.player.rect)

        for ball in self.balls:
            if self.player.laser.collidesWith(ball):
                reward += 1
                new_balls = ball.pop()
                self.player.laser.deactivate()
                if new_balls:
                    self.balls.add(new_balls)
                elif len(self.balls) == 0:
                    self.level += 1
                    self.balls = self.levels.get(self.level)
                    self.balls.update()
            elif pygame.sprite.collide_mask(self.player, ball):
                print("Game over")
                terminated = True

        if self.render_mode == "human":
            self._render_frame()
        else:
            observation = self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        canvas = pygame.Surface((self.width, self.height))
        canvas.fill((0, 0, 0))
        self.player.draw(canvas)
        self.player.laser.draw(canvas)
        for ball in self.balls:
            ball.draw(canvas)

        if self.render_mode == "human":
            self.screen.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.fps)
        else:
            gamestate =  np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            ) # transpose to (1:row, 0:col, 2:channel) from (0:width, 1:height, 2:channel)
            return gamestate

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

    def play(self):
        while(True):
            observation, reward, terminated, truncated, info = self.step()
            if terminated or truncated:
                return
