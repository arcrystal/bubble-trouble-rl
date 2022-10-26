import pygame
from game import TIMESTEP, DISPLAY_HEIGHT

class Laser():
    """
    A pygame object for the game.
    """
    def __init__(self, x):
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