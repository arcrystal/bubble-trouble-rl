import random
import pygame
from direction import Direction
from laser import Laser
from levels import Levels
from agent import Agent
import time
import asyncio


class Game:
    def __init__(self, config):
        self.name = "1D"
        self.steps = 0
        self.render_mode = config.get('render_mode', 'human')
        self.fps = config.get('fps', 60)
        self.window = None
        self.clock = None
        self.width = config.get('width', 720)
        self.height = round(self.width / 1.87)  # 385

        pygame.font.init()
        pygame.init()
        pygame.display.init()

        font = pygame.font.Font('freesansbold.ttf', 32)
        self.win = font.render('Level Complete!', True,
                               (0, 255, 0), (0, 0, 100))
        self.lose = font.render('Game Over...', True,
                                (240, 20, 20), (80, 0, 80))
        self.textRect = self.win.get_rect()
        self.textRect.centerx = self.width / 2
        self.textRect.centery = self.height / 3
        self.window = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.rand_lvl = random.Random()
        self.agent = Agent(self.width / 2, self.height, self.width, self.fps)
        self.agent.laser = Laser(self.width, self.height, self.agent.rect.height, self.fps)
        self.levels = Levels(self.width, self.height, self.fps)
        self.level = self.rand_lvl.randint(1, 7)
        self.balls = self.levels.randomize()
        self._action_to_direction = {
            0: Direction.LEFT,
            1: Direction.RIGHT,
            2: Direction.SHOOT,
            3: Direction.STILL,
        }
        self.reset()

    def reset(self):
        self.steps = 0
        # Reset the player
        self.agent.rect.midbottom = (self.width / 2, self.height)
        self.agent.direction = Direction.STILL
        self.agent.laser.deactivate()
        # Reset the level
        self.level = self.rand_lvl.randint(1, 7)
        self.balls = self.levels.randomize()  # self.levels.get(self.level)
        self._render_frame()

    def step(self, action=None):
        newlevel = False
        gameover = False

        # Update agent, laser, and ball sprites
        keys = pygame.key.get_pressed()
        action = 3
        if keys[pygame.K_LEFT]:
            action = 0
        if keys[pygame.K_RIGHT]:
            action = 1
        if keys[pygame.K_UP]:
            action = 2

        direction = self._action_to_direction[action]
        self.agent.step(direction)
        self.agent.laser.update()
        self.balls.update()

        for ball in self.balls:
            hit = self.agent.laser.collidesWith(ball)
            if hit:
                new_balls = ball.pop()
                self.agent.laser.deactivate()
                if new_balls:
                    self.balls.add(new_balls)

                    if len(self.balls) == 0:
                        self.window.blit(self.win, self.textRect)
                        pygame.display.update()
                        newlevel = True
                        self.balls = self.levels.randomize()
                        self.balls.update()

            elif pygame.sprite.collide_mask(self.agent, ball):
                self.window.blit(self.lose, self.textRect)
                time.sleep(1)

                gameover = True

        self._render_frame()

        return newlevel, gameover

    def _render_frame(self):
        canvas = pygame.Surface((self.width, self.height))
        canvas.fill((0, 0, 0))
        self.agent.laser.draw(canvas)
        self.agent.draw(canvas)
        for ball in self.balls:
            ball.draw(canvas)

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.fps)

    async def play(self):
        while True:
            gameover, newlevel = self.step()
            await asyncio.sleep(0)
            if gameover or newlevel:
                return
