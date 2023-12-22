import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from pprint import pprint
from direction import Direction
from laser import Laser
from levels import Levels
from player import Player


class Game(gym.Env):

    def __init__(self, config):
        self.steps = 0
        self.render_mode = config.get('render_mode', None)
        self.fps = config.get('fps', 60)
        self.rewards = config.get('rewards', {
            'time_passing': 0.0,
            'shoot_when_shooting': 0.0,
            'hit_ceiling': 0.0,
            'hit_ball': 0.0,
            'pop_ball': 0.0,
            'finish_level': 100.0,
            'game_over': -10.0,
            'distance_from_center': 0.0
        })
        self.window = None
        self.clock = None
        self.width = 640
        self.height = 480

        if self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()

        # Initialize game sprites
        self.agent = Player(self.width / 2, self.height, self.width, self.fps)
        self.agent.laser = Laser(self.width, self.height, self.fps)
        self.levels = Levels(self.width, self.height, self.fps)
        self.level = 1
        self.balls = self.levels.get(1)

        self.action_space = gym.spaces.Discrete(4)
        # self.observation_space = spaces.Dict({
        #     "ball_sizes": spaces.Box(low=0.0, high=1.0, shape=(16,)),
        #     "ball_X": spaces.Box(low=0.0, high=1.0, shape=(16,)),
        #     "ball_Y": spaces.Box(low=0.0, high=1.0, shape=(16,)),
        #     "laser_position": spaces.Box(low=0.0, high=1.0, shape=(1,)),
        #     "player_position": spaces.Box(low=0.0, high=1.0, shape=(1,))
        # })
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(50,))

        self._action_to_direction = {
            0: Direction.LEFT,
            1: Direction.RIGHT,
            2: Direction.SHOOT,
            3: Direction.STILL,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset the player
        self.agent.rect.midbottom = (self.width / 2, self.height)
        self.agent.direction = Direction.STILL
        self.agent.laser.deactivate()
        # Reset the level
        self.level = 1
        self.balls = self.levels.get(self.level)
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action=None):
        reward = self.rewards['time_passing']
        terminated = False
        truncated = False

        if self.render_mode == "human" and action is None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True

            keys = pygame.key.get_pressed()
            action = 3
            if keys[pygame.K_LEFT]:
                action = 0
            if keys[pygame.K_RIGHT]:
                action = 1
            if keys[pygame.K_UP]:
                action = 2

        # Handle laser updates and rewards
        laser_active = self.agent.laser.active
        direction = self._action_to_direction[action]

        if laser_active and direction == Direction.SHOOT:
            reward += self.rewards['shoot_when_shooting']

        self.agent.step(direction)
        status = self.agent.laser.update()
        if status == "hit ceiling":
            reward += self.rewards['hit_ceiling']

        #########################
        # LASER SIMULATION REWARD
        #########################
        self.balls.update()
        laser_sim = self.agent.laser._will_collide(self.balls, self.agent.rect.centerx, self.agent.rect.top)
        # Action is [SHOOT]
        if direction == Direction.SHOOT:
            # If we shoot the laser, we should hit a ball
            reward += laser_sim
        # Action is not [SHOOT] but laser is active
        elif self.agent.laser.active:
            # If the laser is already active, it should not matter
            # whether we shoot or not this timestep
            pass
        # Action is not [SHOOT] and laser is not active
        else:
            # If there isn't laser being shot, we should
            # be rewarded if the laser wouldn't hit and
            # penalized if it would hit.
            reward -= laser_sim
        #########################
        #########################

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
                    reward += self.rewards['finish_level']
                    self.level += 1
                    self.balls = self.levels.get(self.level)
                    self.balls.update()
            elif pygame.sprite.collide_mask(self.agent, ball):
                reward += self.rewards['game_over']
                terminated = True

        observation = self._get_obs()
        # reward += (abs(observation['player_position'] - round(self.width / 2))
        #            * self.rewards['distance_from_center'])
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode in ("rgb_array", "human"):
            return self._render_frame()

    def _render_frame(self):
        canvas = pygame.Surface((self.width, self.height))
        canvas.fill((0, 0, 0))
        self.agent.draw(canvas)
        self.agent.laser.draw(canvas)
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
            if terminated or truncated:
                return

    def _get_obs(self):
        ball_sizes = []
        ball_X = []
        ball_Y = []
        for ball in self.balls:
            ball_sizes.append(ball.radius / min(self.width, self.height))
            ball_X.append(ball.rect.centerx / self.width)
            ball_Y.append(ball.rect.centery / self.height)

        for i in range(len(self.balls), 16):
            ball_sizes.append(0.0)
            ball_X.append(0.0)
            ball_Y.append(0.0)

        obs_list = (ball_sizes + ball_X + ball_Y
                    + [self.agent.laser.length / self.height,
                       self.agent.rect.centerx / self.width])
        # obs = {
        #     "ball_sizes": np.array(ball_sizes, dtype=np.float32),
        #     "ball_X": np.array(ball_X, dtype=np.float32),
        #     "ball_Y": np.array(ball_Y, dtype=np.float32),
        #     "laser_position": np.array([self.agent.laser.length / self.height], dtype=np.float32),
        #     "player_position": np.array([self.agent.rect.centerx / self.width], dtype=np.float32)
        # }
        obs = np.array(obs_list, dtype=np.float32)
        assert self.observation_space.contains(obs), "Observation does not match defined space"
        return obs
    def _get_info(self):
        return {}

