import pygame


class Laser(pygame.sprite.Sprite):
    def __init__(self, width, height):
        super().__init__()
        self.width = int(width / 500.0)
        self.image = pygame.Surface((self.width, height), pygame.SRCALPHA)
        self.rect = self.image.get_rect(center=(-9999, -9999))
        self.speed = height / 80
        self.active = False
        self.display_height = height

    def fire(self, x, y):
        self.image.fill((0, 0, 0, 0))
        self.rect.midbottom = (x, self.display_height)
        self.length = self.display_height - y
        self.active = True

    def deactivate(self):
        self.rect.center = (-9999, -9999)
        self.active = False

    def update(self):
        if self.active:
            self.length = min(self.length + self.speed, self.rect.height)
            pygame.draw.rect(self.image, (255, 0, 0),
                             (0, self.rect.height - self.length, self.width, self.length))
            if self.length == self.rect.height:
                self.active = False

    def collidesWith(self, ball):
        if not self.active:
            return False

        ball_center = ball.rect.center
        ball_radius = ball.radius

        if ball_center[1] + ball_radius < (self.rect.bottom - self.length):
            return False

        closest_x = max(self.rect.left, min(ball_center[0], self.rect.right))
        distance_sq = self.square_distance(ball_center[0], 0, closest_x, 0)
        return distance_sq < (ball_radius ** 2)

    def square_distance(self, x1, y1, x2, y2):
        return (x1 - x2) ** 2 + (y1 - y2) ** 2

    def draw(self, screen):
        if self.active:
            screen.blit(self.image, self.rect)
