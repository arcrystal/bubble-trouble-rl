import pygame

class Floor(pygame.sprite.Sprite):
    """
    A pygame object for the platform.
    """
    DISPLAY_WIDTH = 890
    DISPLAY_HEIGHT = DISPLAY_WIDTH / 1.8737
    SPRITES = {0:pygame.image.load("Sprites/platform.png")}

    def __init__(self):
        super().__init__() # equivalent to pygame.sprite.Sprite.__init__(self)
        self.image = Floor.SPRITES[0]
        self.rect = self.image.get_rect()
        self.rect.x = 0
        self.rect.y = Floor.DISPLAY_HEIGHT

    def update(self, t):
        pass