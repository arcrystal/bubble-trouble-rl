import random

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from direction import Direction
from laser import Laser
from levels import Levels
from agent import Agent
import time



class Game(gym.Env):

    def __init__(self, config):
        self.name = "1D"
        self.steps = 0
        self.render_mode = config.get('render_mode', 'human')
        self.fps = config.get('fps', 60)
        self.rewards = {
            'time_passing': 0.0, # every step
            'shoot_when_shooting': -0.0, # every step
            'hit_ball': 0.0, # up to 16x per episode
            'pop_ball': 0.0, # up to 8x per episode
            'finish_level': 10000.0, # up to 8x per episode
            'game_over': -0.0, # once per episode
            'laser_sim': 0.01 # every step
        }
        self.window = None
        self.clock = None
        self.width = config.get('width', 720)
        self.height = round(self.width / 1.87) # 385
        pygame.font.init()
        font = pygame.font.Font('freesansbold.ttf', 32)
        self.win = font.render('Level Complete!', True,
                               (0,255,0), (0,0,100))
        self.lose = font.render('Game Over...', True,
                                (240,20,20), (80,0,80))
        self.textRect = self.win.get_rect()
        self.textRect.centerx = self.width / 2
        self.textRect.centery = self.height / 3

        if self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()

        # Initialize game sprites
        self.rand_lvl = random.Random()
        self.agent = Agent(self.width / 2, self.height, self.width, self.fps)
        self.agent.laser = Laser(self.width, self.height, self.agent.rect.height, self.fps)
        self.levels = Levels(self.width, self.height, self.fps)
        self.level = self.rand_lvl.randint(1,7)
        self.balls = self.levels.randomize() # self.levels.get(self.level)

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(82,))
        self.observation = np.zeros(self.observation_space.shape)

        self._action_to_direction = {
            0: Direction.LEFT,
            1: Direction.RIGHT,
            2: Direction.SHOOT,
            3: Direction.STILL,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        # Reset the player
        self.agent.rect.midbottom = (self.width / 2, self.height)
        self.agent.direction = Direction.STILL
        self.agent.laser.deactivate()
        # Reset the level
        self.level = self.rand_lvl.randint(1,7)
        self.balls = self.levels.randomize() # self.levels.get(self.level)
        self._update_obs()
        if self.render_mode == "human":
            self._render_frame()

        return self.observation, self._get_info()

    def step(self, action=None):
        # self.steps += 1
        # reward = self.steps / 100
        reward = 0
        terminated = False
        truncated = False
        info = {}
        if self.render_mode == "human" and action is None:
            terminated = any(e.type == pygame.QUIT for e in pygame.event.get())

            keys = pygame.key.get_pressed()
            action = 3
            if keys[pygame.K_LEFT]:
                action = 0
            if keys[pygame.K_RIGHT]:
                action = 1
            if keys[pygame.K_UP]:
                action = 2

        # Update agent, laser, and ball sprites
        direction = self._action_to_direction[action]
        self.agent.step(direction)
        self.agent.laser.update()
        self.balls.update()

        # nearest_ball = self.nearest_ball()
        # if abs(nearest_ball) > 0.25 * self.width:
        #     if not self.agent.laser.active:
        #         if direction == Direction.SHOOT:
        #             reward -= 0.01
        #         elif nearest_ball > 0:
        #             if direction == Direction.RIGHT:
        #                 reward += 0.005
        #         else:
        #             if direction == Direction.LEFT:
        #                 reward += 0.005

        for ball in self.balls:
            hit = self.agent.laser.collidesWith(ball)
            if hit:
                new_balls = ball.pop()
                self.agent.laser.deactivate()
                if new_balls:
                    reward += self.rewards['hit_ball']
                    self.balls.add(new_balls)
                else:
                    reward += self.rewards['pop_ball']
                if len(self.balls) == 0:
                    if self.render_mode == 'human':
                        self.window.blit(self.win, self.textRect)
                        pygame.display.update()
                        time.sleep(1)

                    reward += self.rewards['finish_level']
                    self.level += 1
                    truncated = True
                    #self.balls = self.levels.get(self.level)
                    self.balls = self.levels.randomize()
                    self.balls.update()
            elif pygame.sprite.collide_mask(self.agent, ball):
                if self.render_mode == "human":
                    self.window.blit(self.lose, self.textRect)
                    pygame.display.update()
                    time.sleep(1)

                reward += self.rewards['game_over']
                terminated = True

        # After updating balls, simulate laser
        laser_sim = self.agent.laser._will_collide(self.balls, self.agent.rect.centerx)
        shooting = direction == Direction.SHOOT
        if laser_sim and shooting:
            reward += self.rewards["laser_sim"]
        elif (not laser_sim) and (not shooting):
            reward += self.rewards["laser_sim"]
        else:
            reward -= self.rewards["laser_sim"]

        middle = self.width / 2
        if self.agent.rect.centerx < middle - 100 or self.agent.rect.centerx > middle + 100:
            reward -= 9

        self._update_obs()

        if self.render_mode == "human":
            self._render_frame()

        return self.observation, reward, terminated, truncated, info

    def nearest_ball(self):
        dist = self.width
        playerX = self.agent.rect.centerx
        for ball in self.balls:
            d = ball.rect.centerx - playerX
            if abs(d) < abs(dist):
                dist = d

        return dist

    def render(self):
        if self.render_mode in ("rgb_array", "human"):
            return self._render_frame()

    def _render_frame(self):
        canvas = pygame.Surface((self.width, self.height))
        canvas.fill((0, 0, 0))
        self.agent.laser.draw(canvas)
        self.agent.draw(canvas)
        for ball in self.balls:
            ball.draw(canvas)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.fps)
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )  # transpose to (1:row, 0:col, 2:channel) from (0:width, 1:height, 2:channel)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def play(self):
        while (True):
            observation, reward, terminated, truncated, info = self.step()
            if terminated:
                return

    def _get_obs(self):
        if len(self.balls) == 0:
            ball_sizes = []
            ball_X = []
            ball_Y = []
            ball_xspeed = []
            ball_yspeed = []
            for i in range(16):
                ball_sizes.append(0.0)
                ball_X.append(0.0)
                ball_Y.append(0.0)
                ball_xspeed.append(0.0)
                ball_yspeed.append(0.0)
        else:
            ball_sizes = []
            ball_X = []
            ball_Y = []
            ball_xspeed = []
            ball_yspeed = []
            for ball in self.balls:
                ball_sizes.append(ball.radius / min(self.width, self.height))
                ball_X.append(max(min(ball.rect.centerx / self.width, 1), 0))
                ball_Y.append(max(min(ball.rect.centery / self.height, 1), 0))
                ball_xspeed.append(ball.xspeed / self.width)
                ball_yspeed.append(ball.yspeed / self.height)

            n = len(ball_sizes)
            not_filled = n < 16
            while not_filled:
                for ball_idx in range(n):
                    ball_sizes.append(ball_sizes[ball_idx])
                    ball_X.append(ball_X[ball_idx])
                    ball_Y.append(ball_Y[ball_idx])
                    ball_xspeed.append(ball_xspeed[ball_idx]  )
                    ball_yspeed.append(ball_yspeed[ball_idx])
                    if len(ball_sizes) == 16:
                        not_filled = False
                        break

        obs_list = (ball_sizes + ball_X + ball_Y + ball_xspeed + ball_yspeed
                    + [self.agent.laser.length / self.height,
                       self.agent.rect.centerx / self.width])

        return np.array(obs_list).astype(np.float32)

    def _update_obs(self):
        self.observation = self._get_obs()

    def _get_info(self):
        return {key:0 for key, val in self.rewards.items()}

