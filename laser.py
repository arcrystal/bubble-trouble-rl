import pygame
import time
import os

DISPLAY_WIDTH = float(os.environ.get('DISPLAY_WIDTH'))
DISPLAY_HEIGHT = DISPLAY_WIDTH / 1.8737
FPS = float(os.environ.get('FPS'))
TIMESTEP = float(os.environ.get('TIMESTEP'))

class Laser():
    """
    A pygame object for the game.
    """
    MULTIPLIER = 10 * TIMESTEP / 52 * FPS
    SPEED = DISPLAY_HEIGHT / FPS / TIMESTEP
    def __init__(self, x):
        self.time = time.time()
        self.x = x
        self.height = 0
        self.image = pygame.image.load("Sprites/laser.png")
        self.curr = pygame.transform.scale(
            self.image,
            (self.image.get_width(), 5))

        self.rect = self.image.get_rect()
        self.rect.x = x - self.image.get_width() / 2
        self.rect.y = DISPLAY_HEIGHT

    def getX(self):
        return self.x

    def update(self):
        step = DISPLAY_HEIGHT * (time.time() - self.time) * Laser.MULTIPLIER
        self.time = time.time()
        self.height += step
        self.curr = pygame.transform.scale(
            self.image,
            (self.image.get_width(), self.height))
        self.rect = self.image.get_rect()
        self.rect.x = self.x - self.image.get_width() / 2
        self.rect.y = DISPLAY_HEIGHT - self.curr.get_height()

    def hitCeiling(self, t):
        return self.rect.y < Laser.SPEED * t