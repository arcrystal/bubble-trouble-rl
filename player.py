import pygame

from direction import Direction


class Player(pygame.sprite.Sprite):
    def __init__(self, x, y, display_width, fps):
        super().__init__()
        self.laser = None
        self.display_width = display_width
        self.left_sprite = self.load("Sprites/person_left_sm.png")
        self.still_sprite = self.load("Sprites/person_still_sm.png")
        self.right_sprite = self.load("Sprites/person_right_sm.png")
        self.shoot_sprite = self.load("Sprites/person_shoot_sm.png")
        self.image = self.still_sprite
        self.rect = self.image.get_rect()
        self.rect.midbottom = (x, y)
        self.speed = display_width / fps / 2
        self.direction = Direction.STILL

    def load(self, filename):
        image = pygame.image.load(filename)
        return pygame.transform.scale(image, (self.display_width / 40, self.display_width / 16))

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
                self.laser.fire(self.rect.centerx, self.rect.top)

    def update_image(self, direction):
        if direction == Direction.LEFT:
            self.image = self.left_sprite
        elif direction == Direction.RIGHT:
            self.image = self.right_sprite
        elif direction == Direction.SHOOT:
            self.image = self.shoot_sprite
        elif direction == Direction.STILL:
            self.image = self.still_sprite

    def draw(self, window):
        window.blit(self.image, self.rect)
