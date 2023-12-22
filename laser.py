import pygame


class Laser(pygame.sprite.Sprite):
    def __init__(self, width, height, fps):
        super().__init__()
        self.x = -1
        self.y = -1
        self.width = int(width / 500.0)
        self.display_height = height
        self.display_width = width
        self.speed = height / fps
        self.fps = fps
        self.active = False
        self.length = 0.0

    def fire(self, x, y):
        """
        Fire the laser. Now when update is called, the laser will increase in length
        and be drawn to the given screen.
        :param x: The x-coordinate of the agent when the laser is fired. This will
                    be the x-position of the laser for it's firing duration.
        :param y: The y-coordinate of the top of agent. This will remain the y-position
                    of the bottom of the laser on the screen.

        :return: None
        """
        self.x = x
        self.y = y
        self.active = True

    def deactivate(self):
        self.active = False
        self.length = 0

    def update(self):
        if self.active:
            if self.length == self.y:
                self.deactivate()

            self.length = min(self.length + self.speed, self.y)



    def intersects(self, circle_center, circle_radius, line_x):
        x_center, y_center = circle_center

        A = 1.0
        B = 2.0 * x_center
        C = (x_center**2
             + y_center**2
             + line_x**2
             - 2 * y_center * line_x
             - circle_radius**2)

        # Calculate the discriminant
        D = B**2 - 4 * A * C

        # Check the discriminant to determine intersection
        return D >= 0

    def collidesWith(self, ball):
        # lasers is inactive
        if not self.active:
            return False

        ball_x, ball_y = ball.rect.center
        ball_radius = ball.radius

        if (ball_y + ball_radius) < (self.y - self.length):
            # ball is above laser
            return False

        if (ball_x + ball_radius) < self.x:
            # ball is left of laser
            return False

        if (ball_x - ball_radius) > self.x:
            # ball is right of laser
            return False

        if (self.y - self.length < ball_y + ball_radius
            and self.y > ball_y - ball_radius):
            return True

        return False


    def draw(self, canvas):
        if self.active:
            rect = pygame.Rect(self.x, self.y - self.length, self.width, self.length)
            pygame.draw.rect(canvas, (255, 0, 0), rect)

    def copy(self):
        new_laser = Laser(self.width, self.display_height, self.fps)
        new_laser.x = self.x
        new_laser.y = self.y
        new_laser.length = self.length
        new_laser.active = self.active
        return new_laser

    def _will_collide(self, balls):
        if not self.active:
            return 0

        laser_copy = self.copy()
        ball_copies = [ball.copy() for ball in balls]
        while(laser_copy.active):
            laser_copy.update()
            # print(f"\nLaser:", laser_copy)
            for i, ball_copy in enumerate(ball_copies):
                ball_copy.update()
                #print(f"Ball{i+1}:", ball_copy)
                if laser_copy.collidesWith(ball_copy):
                    return 1

        return -1

    def __repr__(self):
        return f"({self.x}, {self.y - self.length}:{self.y})"
