import sys

import pygame.display

from ball import *

RED = (255, 0, 0)
YELLOW = (245, 237, 7)
GREEN = (52, 145, 33)
BLUE = (128, 206, 242)
ORANGE = (237, 141, 45)
PURPLE = (111, 38, 163)

class Levels:
    def __init__(self, width, height, fps):
        self.fps = fps
        self.width = width
        self.height = height
        self.b1bounce = self.width / 10

    def get(self, lvl):
        if lvl == 1:
            balls = [BallLevel2(self.width // 4, self.height // 4, self.width, self.height, BLUE, self.fps)]
        elif lvl == 2:
            balls = [BallLevel3(self.width // 4, self.height // 4, self.width, self.height, GREEN, self.fps)]
        elif lvl == 3:
            balls =  [BallLevel4(self.width // 4, self.height // 4, self.width, self.height, RED, self.fps)]
        elif lvl == 4:
            balls =  [BallLevel3(self.width // 4, self.height // 4, self.width, self.height, ORANGE, self.fps),
                    BallLevel3(3 * self.width // 4, self.height // 4, self.width, self.height, ORANGE, self.fps)]
        elif lvl == 5:
            balls =  [BallLevel3(self.width // 3, self.height // 4, self.width, self.height, YELLOW, self.fps),
                    BallLevel4(2*self.width // 3 - 10, self.height // 4, self.width, self.height, GREEN, self.fps)]
        elif lvl == 6:
            balls =  [BallLevel1(  self.width // 7, 394, self.width, self.height, PURPLE, self.fps),
                    BallLevel1(2*self.width // 7, 394, self.width, self.height, PURPLE, self.fps),
                    BallLevel1(3*self.width // 7, 394, self.width, self.height, PURPLE, self.fps),
                    BallLevel1(4*self.width // 7, 394, self.width, self.height, PURPLE, self.fps),
                    BallLevel1(5*self.width // 7, 394, self.width, self.height, PURPLE, self.fps),
                    BallLevel1(6*self.width // 7, 394, self.width, self.height, PURPLE, self.fps)]
        elif lvl == 7:
            balls =  [BallLevel1(  self.width // 7 - 40, 394,  self.width, self.height, RED, self.fps),
                    BallLevel1(  self.width // 7 - 20, 394,  self.width, self.height, YELLOW, self.fps),
                    BallLevel1(  self.width // 7,      394,  self.width, self.height, ORANGE, self.fps),
                    BallLevel1(2*self.width // 7 - 40, 394,  self.width, self.height, RED, self.fps),
                    BallLevel1(2*self.width // 7 - 20, 394,  self.width, self.height, YELLOW, self.fps),
                    BallLevel1(2*self.width // 7,      394,  self.width, self.height, ORANGE, self.fps),
                    BallLevel1(5*self.width // 7 + 10, 394,  self.width, self.height, RED, self.fps),
                    BallLevel1(5*self.width // 7 + 30, 394,  self.width, self.height, YELLOW, self.fps),
                    BallLevel1(5*self.width // 7 + 50, 394,  self.width, self.height, ORANGE, self.fps),
                    BallLevel1(6*self.width // 7 + 10, 394,  self.width, self.height, RED, self.fps),
                    BallLevel1(6*self.width // 7 + 30, 394,  self.width, self.height, YELLOW, self.fps),
                    BallLevel1(6*self.width // 7 + 50, 394,  self.width, self.height, ORANGE, self.fps)]
        elif lvl == 8:
            balls =  [BallLevel5(self.width // 2, self.height // 4, self.width, self.height, RED, self.fps)]
        else:
            sys.exit(0)

        group = pygame.sprite.Group()
        for ball in balls:
            group.add(ball)

        return group
