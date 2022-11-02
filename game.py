import os
import random 
import matplotlib.pyplot as plt

FPS = float(os.environ.get('FPS'))
DISPLAY_WIDTH = int(os.environ.get('DISPLAY_WIDTH')) # Default 890
TIMESTEP = 1 / FPS
DISPLAY_HEIGHT =int(DISPLAY_WIDTH * 0.5337) # Default 475

import pygame
from player import Player
from barrier import Barrier
from laser import Laser
from levels import BALLS
from itertools import product

VAL_TO_ACTION = {
    0: pygame.K_LEFT,
    1: pygame.K_RIGHT,
    2: pygame.K_UP,
    3: None}

import gym
from gym.spaces import Discrete, Dict, Box

import numpy as np

class Game(gym.Env):
    """
    Game object for falling circles.
    """
    WHITE  = (255, 255, 255)
    BLACK  = (  0,   0,   0)
    RED    = (255,   0,   0)
    GREEN  = (  0, 255,   0)
    BLUE   = (  0,   0, 255)
    ORANGE = (255, 255,   0)
    YELLOW = (  0, 255, 255)
    LVL_TIME = {
        1: 20000,
        2: 35000,
        3: 50000,
        4: 65000,
        5: 80000,
        6: 90000,
        7: 100000,
        8: 100000}

    def __init__(self):
        # https://www.gymlibrary.dev/api/core/#gym.Env.observation_space
        # https://www.gymlibrary.dev/api/core/#gym.Env.action_space
        self.action_space = gym.spaces.Discrete(4)
        self.init_render()

    def init_render(self):
        pygame.init()
        pygame.mixer.init()
        # display_height + platform_height + timer_height
        self.screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT+27+10)) 
        pygame.display.set_caption("Ball Breaker")
        self.level = 1
        self.timer = 0
        self.shooting = False
        self.font = pygame.font.SysFont('Calibri', 25, True, True)
        self.backgrounds = [pygame.image.load("Backgrounds/bluepink.jpg").convert()]
        self.backgrounds *= len(Game.LVL_TIME)
        self.clock = pygame.time.Clock()

    def get_state(self):
        sub_surface = self.screen.subsurface(pygame.Rect(0, 0, DISPLAY_WIDTH, DISPLAY_HEIGHT))
        return pygame.surfarray.array3d(sub_surface)

    # https://www.gymlibrary.dev/api/core/#gym.Env.reset
    def reset(self, mode='rgb', countdown=False):
        """
        Returns:
            info (dict):
                lvlsprites (pygame.sprite.Group): contains all sprites
                player: (pygame.sprite.Sprite): player sprite
                balls (pygame.sprite.Group): contains all ball sprites
                platform: (pygame.sprite.Sprite): platform sprite
                background: (pygame.Surface): background from game screen
                timer (float): keeps track of time elapsed
                timeleft (float): keeps track of time with respect to display size
            observation (object)
        """
        # Reset gameplay variables
        self.timeleft = DISPLAY_WIDTH
        self.background = self.backgrounds[self.level]
        self.timer = 0

        # Creates sprites and convert pixel format to same as final display
        ball_sprites = BALLS[self.level]
        for sprites in ball_sprites:
            for color, sprite in sprites.SPRITES.items():
                sprites.SPRITES[color] = sprite.convert_alpha()

        self.player = Player()
        for key, sprite in self.player.SPRITES.items():
                self.player.SPRITES[key] = sprite.convert_alpha()

        self.platform = Barrier()
        self.platform.image = self.platform.image.convert_alpha()

        # Create sprite groups and add sprites
        self.balls = pygame.sprite.Group()
        self.lvlsprites = pygame.sprite.Group()
        self.balls.add(ball_sprites)
        self.lvlsprites.add(self.player)
        self.lvlsprites.add(self.balls)
        self.lvlsprites.add(self.platform)

        # Render start screen
        if mode=='human':
            lvl_font = self.font.render(f'Level {self.level}', True, Game.GREEN, Game.BLUE)
            lvl_font_rect = lvl_font.get_rect()
            lvl_font_rect.center = DISPLAY_WIDTH / 2, DISPLAY_HEIGHT / 10
            start_ticks=pygame.time.get_ticks()
            pygame.event.get()
            # Draw start countdown:
            while countdown:
                ticks = pygame.time.get_ticks() - start_ticks
                if ticks > 3000:
                    break

                if ticks % 100 == 0:
                    self.screen.blit(self.background, (0, 0))
                    self.lvlsprites.draw(self.screen)
                    text = f"Starting in: {round((3000-ticks)/1000,1)}"
                    render_text = self.font.render(text, True, Game.RED)
                    self.screen.blit(render_text, (DISPLAY_WIDTH / 2 - 10, 75))
                    pygame.display.update()

        return self.get_state()

    # https://www.gymlibrary.dev/api/core/#gym.Env.step
    def step(self, action=None, user=False):
        # Apply action
        if user:
            self.handle_keyevents()
        else:
            self.exit_if_quitting()
            if action == pygame.K_LEFT:
                self.player.left()
            elif action == pygame.K_RIGHT:
                self.player.right()
            elif action == pygame.K_UP and not self.shooting:
                self.shooting = True
                self.laser = Laser(self.player.rect.centerx)
            else:
                if self.player.xspeed != 0:
                    self.player.stop()

        reward = 0
        gameover = False
        for ball in self.balls:
            if pygame.sprite.collide_mask(self.player, ball):
                # print("You lose.")
                self.shooting = False
                gameover = True
                reward = -10
                return self.get_state(), reward, gameover, {}

            if self.shooting:
                self.laser.update()
                if pygame.sprite.collide_mask(self.laser, ball):
                    # print("Laser pop.")
                    self.shooting = False
                    pop_result = ball.pop()
                    self.lvlsprites.remove(ball)
                    self.balls.remove(ball)
                    if pop_result is not None:
                        self.lvlsprites.add(pop_result)
                        self.balls.add(pop_result)
                        reward = 1
                    else:
                        # Get 2.5x points for popping the last ball, so the agent
                        # learns to remove balls from the screen before popping
                        # new ones.
                        reward = 2
                elif self.laser.hitCeiling():
                    # print("Ceiling pop")
                    self.shooting = False

            if pygame.sprite.collide_rect(ball, self.platform):
                ball.bounceY()

        self.clock.tick(FPS)
        timestep = self.clock.get_time()
        self.timer += timestep
        elapsed = DISPLAY_WIDTH / Game.LVL_TIME[self.level] * self.timer
        self.timeleft = DISPLAY_WIDTH - elapsed
        if self.timeleft <= 0:
            # print("Time ran out.")
            # print("You lose.")
            gameover = True
            self.shooting = False
        elif not self.balls:
            self.level += 1
            self.reset()

        self.lvlsprites.update()
        # Draw and update screen
        self.screen.blit(self.background, (0, 0))
        self.draw_timer(self.timeleft)
        self.lvlsprites.draw(self.screen)
        if self.shooting:
            self.screen.blit(self.laser.curr, self.laser.rect)
        
        info = {}
        # observation, reward, truncated, terminated, info
        return self.get_state(), reward, gameover, info

    # https://www.gymlibrary.dev/api/core/#gym.Env.render
    def render(self, mode='rgb'):
        if mode=='human':
            pygame.display.update()

    def close(self):
        pygame.quit()

    def exit_if_quitting(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()

    def handle_keyevents(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.player.left()
                if event.key == pygame.K_RIGHT:
                    self.player.right()
                if event.key == pygame.K_UP and not self.shooting:
                    self.shooting = True
                    self.laser = Laser(self.player.rect.centerx)
                if event.key == pygame.K_i:
                    pygame.image.save(self.screen, "screenshot.png")

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT and self.player.xspeed < 0:
                    self.player.stop()
                if event.key == pygame.K_RIGHT and self.player.xspeed > 0:
                    self.player.stop()

    def draw_timer(self, timeleft):
        pygame.draw.line(
            self.screen, Game.BLACK,
            (0, DISPLAY_HEIGHT+27),
            (timeleft, DISPLAY_HEIGHT+27),
            10)

    def collide(self, laser, ball):
        if laser.rect.x < ball.x + ball.image.get_width() \
            and laser.rect.x + laser.image.get_width() < ball.rect.x:
            if laser.rect.y < ball.rect.y + ball.image.get_height():
                return True

        return False

    def policy(self, observation, mode='train'):
        """
        RL Agent's policy for mapping an observation to an action.

        Args:
            observation: the current state of the environment.
        Returns:
            action (int): pygame global corresponding an action the agent will take.
        Raises:
            None.

        Notes:
        ----------------------
        VAL_TO_ACTION = {
            0: pygame.K_LEFT,
            1: pygame.K_RIGHT,
            2: pygame.K_UP,
            3: None
        }
        """
        if mode == 'train':
            print("Obs shape:", observation.shape)
            action = VAL_TO_ACTION[self.action_space.sample()]
            while self.shooting and action == 2:
                action = VAL_TO_ACTION[self.action_space.sample()]
            
            return action
        
        return None

    def play(self, mode='rbg', num_trials=2):
        """
        Highest level class method for playing or simulating the pygame.

        Args:
            user (bool): True if user is playing, False if simulating with RL agent.
            num_trials: How many trials the game will run.
        Returns:
            None.
        Raises:
            None.
        """
        for trial in range(num_trials):
            gameover = False
            steps = 0
            trial_reward = 0
            observation = self.reset(mode)
            while not gameover:
                action = self.policy(observation, mode)
                observation, reward, gameover, info = self.step(action, mode)
                trial_reward += reward
                steps += 1
                self.render(mode)
            
            # Print statistics and reset the environment
            print("Trial", trial, "reward:", trial_reward)
            self.reset(mode)
                        
        print(info)
        self.close()


# Needs to be observation reward gameover info
# we can make self.level iterate in step and reset takes self.level
# check self.reset in all places.