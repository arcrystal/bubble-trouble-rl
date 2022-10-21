import pygame
import os

DISPLAY_WIDTH = float(os.environ.get('DISPLAY_WIDTH'))
DISPLAY_HEIGHT = DISPLAY_WIDTH / 1.8737

class Floor(pygame.sprite.Sprite):
    """
    A pygame object for the platform.
    """

    def __init__(self):
        super().__init__() # equivalent to pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load("Sprites/platform.png")
        self.rect = self.image.get_rect()
        self.rect.x = 0
        self.rect.y = DISPLAY_HEIGHT

    def update(self):
        pass