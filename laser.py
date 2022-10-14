import pygame
import time

class Laser():
    """
    A pygame object for the game.
    """
    DISPLAY_WIDTH = 890
    DISPLAY_HEIGHT = DISPLAY_WIDTH / 1.8737
    # FPS = 52, TIMESTEP=0.1
    SPEED = DISPLAY_HEIGHT / 5.2 # FPS * TIMESTEP
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
        self.rect.y = Laser.DISPLAY_HEIGHT

    def getX(self):
        return self.x

    def update(self, t):
        step = Laser.DISPLAY_HEIGHT * (time.time() - self.time)
        self.time = time.time()
        self.height += step
        self.curr = pygame.transform.scale(
            self.image,
            (self.image.get_width(), self.height))
        self.rect = self.image.get_rect()
        self.rect.x = self.x - self.image.get_width() / 2
        self.rect.y = Laser.DISPLAY_HEIGHT - self.curr.get_height()

    def hitCeiling(self, t):
        return self.rect.y < Laser.SPEED * t