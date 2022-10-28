import os
FPS = float(os.environ.get('FPS'))
DISPLAY_WIDTH = float(os.environ.get('DISPLAY_WIDTH'))
TIMESTEP = 1 / FPS
DISPLAY_HEIGHT = DISPLAY_WIDTH * 0.5337 # Default 475

import pygame
from player import Player
from floor import Floor
from laser import Laser
from levels import BALLS
#from utils.Utils import getCirclePixels
from itertools import product
def getCirclePixels(centerx, centery, radius):
        for x, y in product(range(int(radius) + 1), repeat=2):
            if x**2 + y**2 <= radius**2:
                yield from set((
                    (centerx + x, centery + y),
                    (centerx + x, centery-y),
                    (centerx-x, centery+y),
                    (centerx-x, centery-y),))

VAL_TO_ACTION = {
    0: pygame.K_LEFT,
    1: pygame.K_RIGHT,
    2: pygame.K_UP,
    3: None}

import gym
from gym.spaces import Discrete, Dict, Box

import numpy as np
from numpy import uint8 as np_uint8
from numpy import array as np_array

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
        self.observation_space = Dict({
            "posX": Discrete(int(DISPLAY_WIDTH)),
            "velX": Discrete(3),
            "balls": Box(
                low=np_array([0., 0.]),
                high=np_array([int(DISPLAY_WIDTH), int(DISPLAY_HEIGHT)]),
                dtype=np_uint8)
        })
        # https://www.gymlibrary.dev/api/core/#gym.Env.action_space
        self.action_space = gym.spaces.Discrete(4)

    def init_render(self):
        pygame.init()
        pygame.mixer.init()
        # display_height + platform_height + timer_height
        self.screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT+27+10)) 
        pygame.display.set_caption("Ball Breaker")
        self.score = 0
        self.level = 1
        self.timer = 0
        self.shooting = False
        self.font = pygame.font.SysFont('Calibri', 25, True, True)
        self.text = self.font.render(f"Score: {self.score}", True, Game.RED)
        self.screen.blit(self.text, (25, 25))
        self.backgrounds = [pygame.image.load("Backgrounds/bluepink.jpg").convert()]
        self.backgrounds *= len(Game.LVL_TIME)
        self.clock = pygame.time.Clock()

    def get_observation(self):
        """
        Users gameplay vars to get an observation of the environment.

        Args:
            None.
        Returns:
            observation tuple(int, int, np.array):
                - int: x position of player
                - int: laser is shooting (at pos x)
                - np.array: array of screen's balls with 0 for empty / size of ball
        Raises:
            ValueError: when the ball is at the edge of the screen, excepts value error
            because ball array is not entirely on the screen
        """
        obs = np.zeros((int(DISPLAY_HEIGHT), int(DISPLAY_WIDTH)), dtype=np.int32)
        for ball in self.balls:
            radius = ball.pixels.shape[0]
            x = int(ball.x)
            y = int(ball.y)
            # TODO: make this more efficient - shouldn't have to update ball.pixels
            # (rn they are pygame pixel values but they should already be ballsize)
            try:
                ball.pixels[np.where(ball.pixels != 0)] = ball.ballsize
                obs[y:y+radius, x:x+radius] = ball.pixels
            except ValueError:
                continue # TODO: actually fix this exception

        # write observation as 2D "string" to file
        # with open("temp.txt", 'w') as f:
        #     for row in obs:
        #         f.write(" ".join(list(row.astype(str))))
        #         f.write("\n")
            
        #     f.close()

        # exit()

        return obs

    # https://www.gymlibrary.dev/api/core/#gym.Env.reset
    def reset(self, lvl, user=False, countdown=False):
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
        self.background = self.backgrounds[lvl]
        self.timer = 0

        # Creates sprites and convert pixel format to same as final display
        ball_sprites = BALLS[lvl]
        for sprites in ball_sprites:
            for color, sprite in sprites.SPRITES.items():
                sprites.SPRITES[color] = sprite.convert_alpha()

        self.player = Player()
        for key, sprite in self.player.SPRITES.items():
                self.player.SPRITES[key] = sprite.convert_alpha()

        self.platform = Floor()
        self.platform.image = self.platform.image.convert_alpha()

        # Create sprite groups and add sprites
        self.balls = pygame.sprite.Group()
        self.lvlsprites = pygame.sprite.Group()
        self.balls.add(ball_sprites)
        self.lvlsprites.add(self.player)
        self.lvlsprites.add(self.balls)
        self.lvlsprites.add(self.platform)

        # Render start screen
        if user:
            lvl_font = self.font.render(f'Level {lvl}', True, Game.GREEN, Game.BLUE)
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

        observation, info = None, {} # TODO: give these information
        return observation, info

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

    # https://www.gymlibrary.dev/api/core/#gym.Env.step
    def step(self, lvl, action=None, user=False):
        # Apply action
        if not user:
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

        # Get collision updates
        nextlevel, gameover = False, False
        for ball in self.balls:
            if pygame.sprite.collide_mask(self.player, ball):
                # print("You lose.")
                self.shooting = False
                observation, reward, gameover, info = 0., 0., True, {}
                return observation, reward, nextlevel, gameover, info

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
                elif self.laser.hitCeiling():
                    # print("Ceiling pop")
                    self.shooting = False

            if pygame.sprite.collide_rect(ball, self.platform):
                ball.bounceY()

        self.clock.tick(FPS)
        self.timer += self.clock.get_time()
        elapsed = DISPLAY_WIDTH / Game.LVL_TIME[lvl] * self.timer
        self.timeleft = DISPLAY_WIDTH - elapsed
        if self.timeleft <= 0:
            # print("Time ran out.")
            # print("You lose.")
            gameover = True
            self.shooting = False
        elif not self.balls:
            nextlevel = True

        self.lvlsprites.update()

        # Get reward
        reward = round(self.score - elapsed / 10)
        # for ball in self.balls: reward -= 2 ** (2*ball.getSize())

        # observation, reward, truncated, terminated, info
        observation, info = None, {}
        return observation, reward, nextlevel, gameover, info

    # https://www.gymlibrary.dev/api/core/#gym.Env.render
    def render(self):
        # Draw and update screen
        self.screen.blit(self.background, (0, 0))
        self.draw_timer(self.timeleft)
        self.lvlsprites.draw(self.screen)
        if self.shooting:
            self.screen.blit(self.laser.curr, self.laser.rect)

        pygame.display.update()

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

    def policy(self, observation):
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
        action = VAL_TO_ACTION[self.action_space.sample()]
        while self.shooting and action == 2:
            action = VAL_TO_ACTION[self.action_space.sample()]
        
        return action

    def play(self, user=False, num_trials=1):
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
        # Set initial game vars
        gameover = False
        nextlevel = False
        self.init_render()
        # Play the game num_trials times
        for trial in range(num_trials):
            gameover = False
            self.score = 0
            # Loop through levels
            for lvl in range(5, 8):
                if gameover:
                    break
                
                # Reset the environment
                observation, _ = self.reset(lvl=lvl, user=user)
                # Start the level
                while not (nextlevel or gameover):
                    if user:
                        # Handle user inputs
                        action = self.handle_keyevents()
                    else:
                        # Exit the game when closing the window.
                        self.exit_if_quitting()
                        # Get an action
                        action = self.policy(observation)
                    
                    # Step the environment forward
                    observation, reward, nextlevel, gameover, _ = self.step(lvl, action, user)
                    self.render()
                    obs = self.get_observation()

                if nextlevel:
                    # Stop lvl from iterating > once
                    nextlevel = False
                    # Update base reward (game score)
                    self.score += Game.LVL_TIME[lvl] / 200
                    self.score += round(self.timeleft / 10)
            
            # Print statistics and reset the environment
            print("Final score:", self.score)
            print("Trial", trial, "reward:", reward)
            self.reset(lvl=1, user=user)
                        
        pygame.quit()