import pygame

class Level(pygame.sprite.Group):
    """
    TODO:
    """
    
    def __init__(self, time, lvl):
        super().__init__(self)
        self.time = time
        self.lvl = lvl
