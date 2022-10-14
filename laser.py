import pygame

class Laser():
    """
    A pygame object for the game.
    """
    DISPLAY_WIDTH = 1280
    DISPLAY_HEIGHT = 960
    SPEED = DISPLAY_HEIGHT / 10
    def __init__(self, x):
        self.x = x
        self.height = 0
        self.image = pygame.image.load("Sprites/laser.png")
        self.curr = pygame.transform.scale(
            self.image,
            (self.image.get_width(), 5))

        self.rect = self.image.get_rect()
        self.rect.x = x - self.image.get_width() / 2
        self.rect.y = 9 * Laser.DISPLAY_HEIGHT / 10 - self.image.get_height()

    def getX(self):
        return self.x

    def update(self, t):
        self.height += Laser.SPEED * t
        self.curr = pygame.transform.scale(
            self.image,
            (self.image.get_width(), self.height))
        self.rect = self.image.get_rect()
        self.rect.x = self.x - self.image.get_width() / 2
        self.rect.y = 9 * Laser.DISPLAY_HEIGHT / 10 - self.curr.get_height()

    def hitCeiling(self, t):
        return self.rect.y < Laser.SPEED * t