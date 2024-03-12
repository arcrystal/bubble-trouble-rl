import pygame

from direction import Direction

LEFT_SPRITE = pygame.image.load("Sprites/left.png")
STILL_SPRITE = pygame.image.load("Sprites/still.png")
RIGHT_SPRITE = pygame.image.load("Sprites/right.png")


class Agent(pygame.sprite.Sprite):
    def __init__(self, x, y, display_width, fps):
        super().__init__()
        self.laser = None
        self.display_width = display_width
        self.left_sprite = self.load(LEFT_SPRITE)
        self.still_sprite = self.load(STILL_SPRITE)
        self.right_sprite = self.load(RIGHT_SPRITE)
        self.image = self.still_sprite
        self.rect = self.image.get_rect()
        self.rect.midbottom = (x, y)
        self.speed = display_width / fps / 5.13
        self.direction = Direction.STILL

    def load(self, image):
        return pygame.transform.scale(image, (
            30 / 720 * self.display_width, 47 / 720 * self.display_width))

    def step(self, direction):
        if not isinstance(direction, Direction):
            raise ValueError("The parameter of step() must be a Direction")

        self.update_image(direction)
        self.direction = direction
        if direction == Direction.LEFT:
            self.rect.x -= self.speed
            self.rect.x = max(0, self.rect.x)
        elif direction == Direction.RIGHT:
            self.rect.x += self.speed
            self.rect.x = min(self.display_width - self.rect.width, self.rect.x)
        elif direction == Direction.SHOOT:
            if not self.laser.active:
                self.laser.fire(self.rect.centerx)

    def update_image(self, direction):
        if direction == Direction.LEFT:
            self.image = self.left_sprite
        elif direction == Direction.RIGHT:
            self.image = self.right_sprite
        elif direction == Direction.STILL or direction == Direction.SHOOT:
            self.image = self.still_sprite

    def draw(self, window):
        window.blit(self.image, self.rect)
