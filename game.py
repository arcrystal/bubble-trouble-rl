
import os
import random

import gym
import numpy as np
import math

FPS = float(os.environ.get('FPS'))
DISPLAY_WIDTH = int(os.environ.get('DISPLAY_WIDTH')) # Default 890
TIMESTEP = 1 / FPS
DISPLAY_HEIGHT = int(DISPLAY_WIDTH * 0.5337) # Default 475
GAMESTATE = np.zeros((16,32,3), np.int8)
RATIOS = {'y':DISPLAY_HEIGHT/16, 'x':DISPLAY_WIDTH/32}

import pygame
from player import Player
from barrier import Barrier
from laser import Laser
from levels import Levels

VAL_TO_ACTION = {
    0: pygame.K_LEFT,
    1: pygame.K_RIGHT,
    2: pygame.K_UP,
    3: None}


class Game(gym.Env):
    """
    Game object for falling circles.
    """
    WHITE  = (255, 255, 255)
    BLACK  = (  0,   0,   0)
    RED    = (255,   0,   0)
    GREEN  = (  0, 255,   0)
    BLUE  = (  0,   0, 255)
    PINK   = (255, 192, 203)
    # Take out two 0's
    LVL_TIME = {
        1: 2000000,
        2: 3500000, 
        3: 5000000,
        4: 6500000,
        5: 8000000,
        6: 9000000,
        7: 10000000,
        8: 10000000}
    LEVELS = Levels()
    

    def __init__(self, training=True, model_type='dense', model=None, visualize=True, n_features=43, frames=None):
        # https://www.gymlibrary.dev/api/core/#gym.Env.observation_space
        # https://www.gymlibrary.dev/api/core/#gym.Env.action_space
        self.action_space = gym.spaces.Discrete(4)
        self.init_render(training)
        self.model_type = model_type
        self.model = model
        self.visualize = visualize
        self.n_features = n_features
        self.steps = 0
        self.frames = frames
        self.action = [False, False, False]

    def init_render(self, training):
        pygame.init()
        pygame.mixer.init()
        self.screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT+27+10))
        # display_height + platform_height + timer_height
        if not training:
            self.backgrounds = [pygame.image.load("Backgrounds/bluepink.jpg").convert()]
            self.backgrounds *= len(Game.LVL_TIME)
            pygame.display.set_caption("Ball Breaker")
            self.font = pygame.font.SysFont('Calibri', 25, True, True)

        self.timer = 0
        self.level = 1
        self.shooting = False
        self.clock = pygame.time.Clock()
    
    def get_state(self, type='dense'):
        if type=='dense':
            x = (self.player.getX() + self.player.getWidth() / 2) / DISPLAY_WIDTH
            features = [x, int(self.shooting)]
            for ball in self.balls:
                features += ball.get_features()
            
            return features + [0]*(self.n_features-len(features))
        elif type=='conv':
            pixel_data = pygame.surfarray.array2d(self.screen)
            greyscale = np.dot(pixel_data[..., :3], [0.2989, 0.5870, 0.1140])
            resized_array = np.resize(greyscale, (16, 32))
            return resized_array
        else:
            print("unknown model type")
            exit()
        
        if self.model_type == 'conv':
            pixel_data = pygame.surfarray.array2d(self.screen)
            greyscale = np.dot(pixel_data[..., :3], [0.2989, 0.5870, 0.1140])
            resized_array = np.resize(greyscale, (42, 84))
            return resized_array
        
    def mirror(self, state):
        if self.model_type == 'dense':
            mirr = [1-feature for feature in state]
            mirr[2] = state[2]
            return mirr
        elif self.model_type == 'conv':
            return np.flip(GAMESTATE, axis=1)

    # https://www.gymlibrary.dev/api/core/#gym.Env.reset
    def reset(self, mode='rgb', countdown=False, lvl_complete=False):
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
            observation (np.array): 3D array of the screen at the current timestep
        """
        # Reset gameplay variables
        self.timeleft = DISPLAY_WIDTH
        self.timer = 0

        # Creates sprites
        ball_sprites = Game.LEVELS.get(self.level)
        self.player = Player()
        self.platform = Barrier()

        # Convert alphas so sprites have the pixel format as display
        for sprites in ball_sprites:
            for color, sprite in sprites.SPRITES.items():
                sprites.SPRITES[color] = sprite.convert_alpha()
        for key, sprite in self.player.SPRITES.items():
            self.player.SPRITES[key] = sprite.convert_alpha()
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
            self.background = self.backgrounds[self.level]
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
        elif mode=='rgb':
            if lvl_complete:
                self.level -= 1

            self.level %= 5
            self.level += 1
            self.screen.fill((0, 0, 0))
            self.lvlsprites.draw(self.screen)

        return self.get_state()

    # https://www.gymlibrary.dev/api/core/#gym.Env.step
    def step(self, action=None, mode='rgb'):
        """
        RL Agent's step function.

        Args:
            action (int): pygame global corresponding an action the agent will take.
            mode (str): render mode
        Returns:
            observation: the current state of the environment.
            reward: the reward of taking a certain action in the current state
            gameover (bool): if the player loses
        Raises:
            None.

        Notes:
            The reward function has trouble telling the network how to behave and needs work.
        """
        reward = -0.001
        self.steps += 1
        if action == None:
            # Handle key events when player is playing
            self.handle_keyevents()
        else:
            direction = None
            # self.exit_if_quitting() # wastes time but you can close the window
            if action in (pygame.K_LEFT, 0):
                direction = self.player.left()
            elif action in (pygame.K_RIGHT, 1):
                direction = self.player.right()
            elif action in (pygame.K_UP, 2):
                if self.shooting:
                    pass
                else:
                    self.shooting = True
                    self.laser = Laser(self.player.rect.centerx)
            else:
                if self.player.xspeed != 0:
                    self.player.stop()
            if self.player.bad_move(direction):
                pass

        # Discourage spending time
        gameover = False
        for ball in self.balls:
            if ball.rect.y + 100 > self.player.getY():
                if pygame.sprite.collide_mask(self.player, ball):
                    self.shooting = False
                    gameover = True
                    reward = -1
                    info = {'action':action}
                    return self.get_state(), reward, gameover, info
            if self.shooting:
                if self.laser.collideobjects([ball]):
                    self.shooting = False
                    self.laser.hitBall()
                    pop_result = ball.pop()
                    self.lvlsprites.remove(ball)
                    self.balls.remove(ball)
                    if pop_result is not None:
                        self.lvlsprites.add(pop_result)
                        self.balls.add(pop_result)

            if pygame.sprite.collide_rect(ball, self.platform):
                ball.bounceY()

            if ball.x < 0 or ball.x > DISPLAY_WIDTH - ball.width:
                ball.bounceX()

        # Update sprites
        if self.shooting:
            self.laser.update()
            if self.laser.hitCeiling():
                self.shooting = False

        self.lvlsprites.update()
        if mode == 'human':
            self.clock.tick(FPS)
            timestep = self.clock.get_time()
            self.timer += timestep
            elapsed = DISPLAY_WIDTH / Game.LVL_TIME[self.level] * self.timer
            self.timeleft = DISPLAY_WIDTH - elapsed
            if self.timeleft <= 0:
                gameover = True
                self.shooting = False
                
            self.screen.blit(self.background, (0, 0))
            self.draw_timer(self.timeleft)
            self.lvlsprites.draw(self.screen)
            if self.shooting:
                pygame.draw.rect(self.screen, Game.PINK, self.laser)
        elif mode == 'rgb':
            self.screen.fill((0, 0, 0))
            self.lvlsprites.draw(self.screen)
            if self.shooting:
                pygame.draw.rect(self.screen, Game.PINK, self.laser)

        # Level Complete
        if not self.balls:
            reward = 1
            self.level += 1
            self.reset(mode, lvl_complete=True)
        info = {'action':action}
        return self.get_state(), reward, gameover, info

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
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT and self.player.xspeed < 0:
                    self.player.stop()
                    self.action[0] = False
                if event.key == pygame.K_RIGHT and self.player.xspeed > 0:
                    self.player.stop()
                    self.action[1] = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.player.left()
                    self.action[0] = True
                elif event.key == pygame.K_RIGHT:
                    self.player.right()
                    self.action[1] = True
                elif event.key == pygame.K_UP:
                    if not self.shooting:
                        self.action[2] = True
                        self.shooting = True
                        self.laser = Laser(self.player.rect.centerx)
                elif event.key == pygame.K_i:
                    pygame.image.save(self.screen, "screenshot.png")

    def draw_timer(self, timeleft):
        pygame.draw.line(
            self.screen, Game.RED,
            (0, DISPLAY_HEIGHT+27),
            (timeleft, DISPLAY_HEIGHT+27),
            10)

    def repeat(self, X):
        assert len(X) >= self.frames, "not enough training examples"
        return [X[i-self.frames:i] for i in range(self.frames, len(X))]

    def play(self, mode='rgb', num_trials=1, save_data=False, model=None):
        """
        Highest level class method for playing or simulating the pygame.

        Args:
            mode (str): 'human' if user is playing, 'rgb' if simulating with RL agent.
            num_trials: How many trials the game will run.
        Returns:
            None.
        Raises:
            None.
        Notes:
            VAL_TO_ACTION = {
                0: pygame.K_LEFT,
                1: pygame.K_RIGHT,
                2: pygame.K_UP,
                3: None
            }
        """
        FRAMES = math.ceil(DISPLAY_HEIGHT / math.floor(DISPLAY_HEIGHT * TIMESTEP)) + 1
        n_features = 43
        self.init_render(False)
        if save_data or model:
            observations = []
            observations_mirror = []
            actions = []
            actions_mirror = []
        for _ in range(num_trials):
            gameover = False
            observation = self.reset(mode)
            steps = 0
            while not gameover:
                steps += 1
                if model and steps > FRAMES:
                    print("Test")
                    in_frames = np.array(observations[-FRAMES:]).reshape(1, FRAMES, n_features)
                    action = model.predict(in_frames).argmax()
                else:
                    action = None
                obs, reward, gameover, info = self.step(action=action, mode=mode)
                if save_data or model:
                    observations.append(obs)
                    observations_mirror.append(self.mirror(obs))
                    if any(self.action):
                        a = self.action.index(True)
                    else:
                        a = 3

                    actions.append(a)
                    if a==0:
                        actions_mirror.append(1)
                    elif a==1:
                        actions_mirror.append(0)
                    elif a==2:
                        actions_mirror.append(a)
                        self.action[2] = False
                    elif a==3:
                        actions_mirror.append(a)                    
                self.render(mode)
            self.reset(mode)

        self.close()
        pygame.display.quit()

        if save_data:
            cutoff = None
            while not isinstance(cutoff, float):
                try:
                    # cutoff = float(input("Input proportion of gameplay to use for training (float): "))
                    cutoff = 0.9
                except ValueError:
                    pass

            cutoff = int(len(observations)*cutoff)
            # Yreg = actions[self.frames:cutoff]
            # Ymirror = actions_mirror[self.frames:cutoff]
            # Y = np.array(Yreg+Ymirror)
            Y = actions[self.frames:cutoff]
            try:
                # Xreg = np.array(self.repeat(observations[:cutoff]))
                # Xmirror = np.array(self.repeat(observations_mirror[:cutoff]))
                # X = np.concatenate([np.array(Xreg), np.array(Xmirror)], axis=0)
                X = np.array(self.repeat(observations[:cutoff]))
            except AssertionError:
                print("No training performed.")
                return
        
            Y = to_categorical(Y)
            np.save('X.npy', X)
            np.save('Y.npy', Y)

        return

    def test(self, mode='rgb', num_trials=1):
        self.init_render(False)
        for _ in range(num_trials):
            gameover = False
            obs = self.reset(mode)
            obs_multi = np.zeros((self.frames, self.n_features))
            step = 0
            while not gameover:
                if step<self.frames:
                    obs_multi[step] = np.array(obs)
                    action = random.randint(0,3)
                    step += 1
                else:
                    np.roll(obs_multi, -1, axis=1)
                    obs_multi[-1] = obs
                    input = obs_multi.reshape((1,)+obs_multi.shape)
                    action = self.model.predict(input).argmax()
                obs, reward, gameover, info = self.step(action=action, mode=mode)
                self.render(mode)
            
            self.reset(mode)
        
        self.close()
