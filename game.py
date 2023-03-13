import os
import random

FPS = float(os.environ.get('FPS'))
DISPLAY_WIDTH = int(os.environ.get('DISPLAY_WIDTH')) # Default 890
TIMESTEP = 1 / FPS
DISPLAY_HEIGHT = int(DISPLAY_WIDTH * 0.5337) # Default 475

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

import gym

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
    PINK   = (255, 192, 203)
    LVL_TIME = {
        1: 20000,
        2: 35000,
        3: 50000,
        4: 65000,
        5: 80000,
        6: 90000,
        7: 100000,
        8: 100000}
    LEVELS = Levels()

    def __init__(self, training=True, model=None, visualize=True, n_features=128):
        # https://www.gymlibrary.dev/api/core/#gym.Env.observation_space
        # https://www.gymlibrary.dev/api/core/#gym.Env.action_space
        self.action_space = gym.spaces.Discrete(4)
        self.init_render(training)
        self.model = model
        self.visualize = visualize
        self.n_features = n_features

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

    def get_state(self):
        features = [self.player.getX(), self.player.getY(), int(self.shooting)]
        for ball in self.balls:
            # xpos, ypos, xspeed, yspeed, size
            features += ball.get_features()
        
        return features + [0]*(self.n_features-len(features))

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
            observation (np.array): 3D array of the screen at the current timestep
        """
        if not self.visualize:
            mode = 'fast'
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
            self.level = 1
            self.screen.fill((0, 0, 0))
            self.lvlsprites.draw(self.screen)
        else:
            self.level = 1

        return self.get_state()

    # https://www.gymlibrary.dev/api/core/#gym.Env.step
    def step(self, action=None, mode='rgb'):
        reward = 0
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
                    reward -= 99
                else:
                    self.shooting = True
                    self.laser = Laser(self.player.rect.centerx)
            else:
                if self.player.xspeed != 0:
                    self.player.stop()
            if self.player.bad_move(direction):
                reward -= 99

        # Discourage spending time
        reward -= 0.001
        gameover = False
        for ball in self.balls:
            if ball.rect.y + 100 > self.player.getY():
                if pygame.sprite.collide_mask(self.player, ball):
                    self.shooting = False
                    gameover = True
                    reward -= 99
                    return self.get_state(), reward, gameover, {}

            if self.shooting:
                self.laser.update()
                if self.laser.collideobjects([ball]):
                    # print("Laser pop.")
                    reward += 0.25
                    self.shooting = False
                    pop_result = ball.pop()
                    self.lvlsprites.remove(ball)
                    self.balls.remove(ball)
                    if pop_result is not None:
                        self.lvlsprites.add(pop_result)
                        self.balls.add(pop_result)
                elif self.laser.hitCeiling():
                    reward -= 99
                    self.shooting = False

            if pygame.sprite.collide_rect(ball, self.platform):
                ball.bounceY()
            if ball.x < 0 or ball.x > DISPLAY_WIDTH - ball.width:
                ball.bounceX()
        # Level Complete
        if not self.balls:
            self.level += 1
            reward += 99
            self.reset(mode)

        # Update sprites
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
                #self.screen.blit(self.laser.curr, self.laser.rect)
        elif mode == 'rgb':
            self.screen.fill((0, 0, 0))
            self.lvlsprites.draw(self.screen)
            if self.shooting:
                pygame.draw.rect(self.screen, Game.PINK, self.laser)
                #self.screen.blit(self.laser.curr, self.laser.rect)
        
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
            self.screen, Game.RED,
            (0, DISPLAY_HEIGHT+27),
            (timeleft, DISPLAY_HEIGHT+27),
            10)

    # def collide(self, laser, ball):
    #     if laser.rect.x < ball.x + ball.image.get_width() \
    #         and laser.rect.x + laser.image.get_width() < ball.rect.x:
    #         if laser.rect.y < ball.rect.y + ball.image.get_height():
    #             return True

    #     return False

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
            if random.random() > 0.99 or self.model==None:
                action = VAL_TO_ACTION[self.action_space.sample()]
                while self.shooting and action == 2:
                    action = VAL_TO_ACTION[self.action_space.sample()]                
            else:
                while self.shooting and action == 2:
                    action = self.model.predict(observation)

            return action
        
        return None

    def play(self, mode='rgb', num_trials=2):
        """
        Highest level class method for playing or simulating the pygame.

        Args:
            mode (str): 'human' if user is playing, 'rgb' if simulating with RL agent.
            num_trials: How many trials the game will run.
        Returns:
            None.
        Raises:
            None.
        """
        self.init_render(False)
        for _ in range(num_trials):
            gameover = False
            observation = self.reset(mode)
            while not gameover:
                action = self.policy(observation, mode)
                observation, _, gameover, _ = self.step(action, mode)
                self.render(mode)
            
            self.reset(mode)
                        
        self.close()
