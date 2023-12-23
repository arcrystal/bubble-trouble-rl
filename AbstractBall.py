import pygame


class AbstractBall(pygame.sprite.Sprite):
    def __init__(self, x, y, display_width, display_height, color, fps=36, right=True):
        super().__init__()
        self.color = color
        radius, max_yspeed, yacc = self.load_properties(display_width)
        self.radius = radius
        self.fps = fps
        self.x = x
        self.y = y
        self.xspeed = display_width / 9.43333 * (1 if right else -1)
        self.yspeed = 0
        self.max_yspeed = max_yspeed
        self.yacc = yacc
        self.rect = pygame.Rect(x, y, radius*2, radius*2)
        self.timestep = 1.0 / fps
        self.display_width = display_width
        self.display_height = display_height
        surface = pygame.Surface((display_width, display_height), pygame.SRCALPHA)
        pygame.draw.circle(surface, color, (self.radius, self.radius), self.radius)
        self.mask = pygame.mask.from_surface(surface)

    def load_properties(self, display_width):
       return 0, 0, 0

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
            self.yspeed = -self.max_yspeed  # Reverse the vertical speed
            self.rect.bottom = self.display_height

        # Update speed
        self.yspeed = self.clip(self.yspeed + self.yacc * self.timestep)

    def clip(self, speed):
        if speed > 0:
            return min(speed, self.max_yspeed)
        else:
            return max(speed, -self.max_yspeed)

    def draw(self, window):
        pygame.draw.circle(window, self.color, self.rect.center, self.radius)

    def copy(self):
        return type(self)(self.x, self.y, self.display_width,
                            self.display_height, self.color, self.fps)

    def __repr__(self):
        return f"({self.x}, {self.y}), {self.radius})"
