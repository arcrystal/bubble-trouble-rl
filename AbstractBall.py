from abc import ABC, abstractmethod

import pygame


class AbstractBall(pygame.sprite.Sprite, ABC):
    def __init__(self, x, y, display_width, display_height, color, fps=36, right=True):
        super().__init__()
        self.color = color
        size, yacc, yspeed = self.load_properties(display_width, display_height)
        self.radius = size // 2
        self.xspeed = display_width / (9.5 if right else -9.5)
        self.xacc = 1.5
        self.yspeed = yspeed
        self.yacc = yacc
        self.rect = pygame.Rect(x, y, size, size)
        self.fps = fps
        self.timestep = 1 / fps
        self.display_width = display_width
        self.display_height = display_height
        surface = pygame.Surface((display_width, display_height), pygame.SRCALPHA)
        pygame.draw.circle(surface, color, (self.radius, self.radius), self.radius)
        self.mask = pygame.mask.from_surface(surface)

    @abstractmethod
    def load_properties(self, display_width, display_height):
        pass

    @abstractmethod
    def pop(self):
        pass

    def update(self):
        self.rect.x += self.xspeed * self.timestep

        # Check for wall collision
        if self.rect.left < 0:
            self.xspeed = -self.xspeed
            self.rect.left = -self.rect.left
        elif self.rect.right > self.display_width:
            self.xspeed = -self.xspeed
            self.rect.right = self.display_width - (self.rect.right - self.display_width)

        # Update vertical movement
        self.rect.y += self.yspeed * self.timestep + 0.5 * self.yacc * self.timestep ** 2

        # Check for ceiling collision (pop)
        if self.rect.top < 0:
            self.pop()
            return

        # Check for floor collision
        if self.rect.bottom > self.display_height:
            self.rect.bottom = self.display_height - (self.rect.bottom - self.display_height)
            self.yspeed = -self.yspeed  # Reverse the vertical speed

        # Update speed
        self.xspeed += self.xacc * self.timestep
        self.yspeed += self.yacc * self.timestep

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, self.rect.center, self.radius)
