import pygame
import time
import os

FPS = float(os.environ.get('FPS'))
TIMESTEP = 1 / FPS
DISPLAY_WIDTH = float(os.environ.get('DISPLAY_WIDTH'))
DISPLAY_HEIGHT = DISPLAY_WIDTH * 0.5337 # Default 475

class Laser():
    """
    A pygame object for the game.
    """
    # STEP = DISPLAY_HEIGHT / FPS / TIMESTEPß
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
        self.height += DISPLAY_HEIGHT * TIMESTEP
        self.curr = pygame.transform.scale(
            self.image,
            (self.image.get_width(), self.height))
        self.rect = self.image.get_rect()
        self.rect.x = self.x - self.image.get_width() / 2
        self.rect.y = DISPLAY_HEIGHT - self.curr.get_height()

    def hitCeiling(self):
        return self.rect.y < 0