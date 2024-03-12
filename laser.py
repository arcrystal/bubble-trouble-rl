import pygame


class Laser(pygame.sprite.Sprite):
    def __init__(self, width, height, agent_height, fps):
        super().__init__()
        self.x = 0.0
        self.agent_height = agent_height
        self.width = int(width / 500.0)
        self.display_height = height
        self.display_width = width
        self.speed = height / fps
        self.fps = fps
        self.active = False
        self.length = 0.0

    def fire(self, x):
        """
        Fire the laser. Now when update is called, the laser will increase in length
        and be drawn to the given screen.
        :param x: The x-coordinate of the agent when the laser is fired. This will
                    be the x-position of the laser for it's firing duration.

        :return: None
        """
        self.x = x
        self.active = True

    def deactivate(self):
        self.active = False
        self.length = self.agent_height

    def update(self):
        if self.active:
            if self.length >= self.display_height:
                self.deactivate()

            self.length = min(self.length + self.speed, self.display_height)

    def collides_with(self, ball):
        # lasers is inactive
        if not self.active:
            return False

        ball_x, ball_y = ball.rect.center
        ball_radius = ball.radius

        if (ball_y + ball_radius) < (self.display_height - self.length):
            # ball is above laser
            return False

        if (ball_x + ball_radius) < self.x:
            # ball is left of laser
            return False

        if (ball_x - ball_radius) > self.x:
            # ball is right of laser
            return False

        if (self.display_height - self.length < ball_y + ball_radius
                and self.display_height > ball_y - ball_radius):
            return True

        return False

    def draw(self, canvas):
        if self.active:
            rect = pygame.Rect(self.x, self.display_height - self.length, self.width, self.length)
            pygame.draw.rect(canvas, (255, 0, 0), rect)

    def copy(self):
        new_laser = Laser(self.width, self.display_height, self.agent_height, self.fps)
        new_laser.x = self.x
        new_laser.length = self.length
        new_laser.active = self.active
        return new_laser

    def _will_collide(self, balls, x=None):
        laser_copy = self.copy()
        if not laser_copy.active:
            laser_copy.deactivate()
            laser_copy.fire(x)

        ball_copies = [ball.copy() for ball in balls]
        while laser_copy.active:
            laser_copy.update()
            for ball_copy in ball_copies:
                ball_copy.update()
                if laser_copy.collides_with(ball_copy):
                    return True

        return False

    def __repr__(self):
        return f"({self.x}, {self.display_height - self.length}:{self.display_height})"
