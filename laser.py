import pygame
from game import TIMESTEP, DISPLAY_HEIGHT, GAMESTATE, RATIOS
import math

class Laser(pygame.Rect):
    """
    A pygame object for the game.
    """
    def __init__(self, x):
        super().__init__(x-1, DISPLAY_HEIGHT, 2, 0)
        self.yspeed = math.floor(DISPLAY_HEIGHT * TIMESTEP)
        self.gsYStep = self.yspeed / RATIOS['y']
        self.gsX = min(round(x / RATIOS['x']), 31)
        self.gsY = GAMESTATE.shape[0]
        GAMESTATE[-1, self.gsX, 2] = 1

    def getX(self):
        return self.x

    def update(self):
        self.top -= self.yspeed
        self.height += self.yspeed
        self.gsY -= self.gsYStep
        GAMESTATE[round(self.gsY):, self.gsX, 2] = 1

    def hitCeiling(self):
        if self.top < 0:
            GAMESTATE[:, self.gsX, 2] = 0
            return True
        return False
    
    def hitBall(self):
        GAMESTATE[:, self.gsX, 2] = 0