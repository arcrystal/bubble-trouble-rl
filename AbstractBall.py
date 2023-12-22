import pygame


class AbstractBall(pygame.sprite.Sprite):
    def __init__(self, x, y, display_width, display_height, color, fps=36, right=True):
        super().__init__()
        self.color = color
        size, yacc, yspeed, ballLevel = self.load_properties(display_width, display_height)
        self.ballLevel = ballLevel
        self.radius = size // 2
        self.fps = fps
        self.x = x
        self.y = y
        self.xspeed = display_width / fps * (9.5 if right else -9.5)
        self.yspeed = yspeed / fps * 36
        self.yacc = yacc
        self.rect = pygame.Rect(x, y, size, size)
        self.timestep = 1.0 / fps
        self.display_width = display_width
        self.display_height = display_height
        surface = pygame.Surface((display_width, display_height), pygame.SRCALPHA)
        pygame.draw.circle(surface, color, (self.radius, self.radius), self.radius)
        self.mask = pygame.mask.from_surface(surface)

    def load_properties(self, display_width, display_height):
        # defaults to level 1 ball
        size = int(display_width / 50.7273)
        yacc = display_height
        bounce_time = 27.17 / 25 / 2
        yspeed = yacc * bounce_time / 100
        return size, yacc, yspeed, 1

    def pop(self):
        pass

    def update(self):
        self.x += self.xspeed * self.timestep
        self.rect.x = round(self.x)

        # Check for wall collision
        if self.rect.left < 0:
            self.xspeed = -self.xspeed
            self.rect.left = -self.rect.left
        elif self.rect.right > self.display_width:
            self.xspeed = -self.xspeed
            self.rect.right = self.display_width - (self.rect.right - self.display_width)

        # Update vertical movement
        self.y += self.yspeed * self.timestep + 0.5 * self.yacc * self.timestep ** 2
        self.rect.y = round(self.y)
        # Check for ceiling collision (pop)
        if self.rect.top < 0:
            self.pop()
            return

        # Check for floor collision
        if self.rect.bottom > self.display_height:
            self.yspeed = -self.yspeed  # Reverse the vertical speed
            self.rect.bottom = self.display_height - (self.rect.bottom - self.display_height)

        # Update speed
        self.yspeed += self.yacc * self.timestep

    def draw(self, window):
        pygame.draw.circle(window, self.color, self.rect.center, self.radius)

    def copy(self):
        return type(self)(self.x, self.y, self.display_width,
                            self.display_height, self.color, self.fps)

    def __repr__(self):
        return f"({self.x}, {self.y}), {self.radius})"
