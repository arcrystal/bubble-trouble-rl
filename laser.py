import pygame
from game import TIMESTEP, DISPLAY_HEIGHT
import math

class Laser(pygame.Rect):
    """
    A pygame object for the game.
    """
    def __init__(self, x):
        super().__init__(x-1, DISPLAY_HEIGHT, 2, 0)
        self.yspeed = math.floor(DISPLAY_HEIGHT * TIMESTEP)

    def getX(self):
        return self.x

    def update(self):
        self.top -= self.yspeed
        self.height += self.yspeed

    def hitCeiling(self):
        return self.top < 0