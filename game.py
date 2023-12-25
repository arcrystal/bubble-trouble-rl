import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from direction import Direction
from laser import Laser
from levels import Levels
from player import Player


class Game(gym.Env):

    def __init__(self, config):
        self.steps = 0
        self.print_stats = config.get('print_stats', False)
        self.render_mode = config.get('render_mode', 'human')
        self.fps = config.get('fps', 60)
        self.rewards = config.get('rewards', {
            'time_passing': 0.0,
            'shoot_when_shooting': 0.0,
            'hit_ceiling': 0.0,
            'hit_ball': 0.0,
            'pop_ball': 4.0,
            'finish_level': 100.0,
            'game_over': -20.0,
            'nearest_ball': -0.001
        })
        self.width = config.get('width', 720)
        self.window = None
        self.clock = None
        self.height = round(self.width / 1.87) # 385

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

        self.balls.update()
        distanceReward = self.nearestBall() * self.rewards['nearest_ball']
        reward += distanceReward
        laser_sim = self.agent.laser._will_collide(self.balls, self.agent.rect.centerx, self.agent.rect.top)
        if direction == Direction.SHOOT:
            # If we shoot the laser, we should hit a ball
            reward += laser_sim
        elif not self.agent.laser.active:
            # If there isn't laser being shot, we should
            # penalized if it would hit.
            reward -= max(laser_sim, 0)

        if self.print_stats:
            print("Nearest ball:", distanceReward)
            print("Laser sim:", reward - distanceReward)


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
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def nearestBall(self):
        dist = self.width
        playerX = self.agent.rect.centerx
        for ball in self.balls:
            d1 = abs(ball.rect.x - playerX)
            d2 = abs(ball.rect.centerx + ball.radius - playerX)
            d = min(d1, d2)
            if d < dist:
                dist = d

        return dist - 50 # if player is close to ball, reward will be positive

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
            ball_Y.append(max(ball.rect.centery / self.height, 0))

        for i in range(len(self.balls), 16):
            ball_sizes.append(0.0)
            ball_X.append(0.0)
            ball_Y.append(0.0)

        obs_list = (ball_sizes + ball_X + ball_Y
                    + [self.agent.laser.length / self.height,
                       self.agent.rect.centerx / self.width])

        obs = np.array(obs_list, dtype=np.float32)
        if not self.observation_space.contains(obs):
            print(f"Observation space does not contain this observation:\n{obs.tolist()}")

        return obs
    def _get_info(self):
        return {}

