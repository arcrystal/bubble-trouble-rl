
from AbstractBall import AbstractBall

RED = (255, 0, 0)
YELLOW = (245, 237, 7)
GREEN = (52, 145, 33)
BLUE = (128, 206, 242)
ORANGE = (237, 141, 45)

class BallLevel1(AbstractBall):
    def load_properties(self, display_width, display_height):
        size = int(display_width / 50.7273)
        yacc = display_height
        bounce_time = 27.17 / 25 / 2
        yspeed = yacc * bounce_time / 100
        return size, yacc, yspeed, 1

    def pop(self):
        self.kill()


class BallLevel2(AbstractBall):
    def load_properties(self, display_width, display_height):
        size = int(display_width / 42.7273)
        yacc = display_height
        bounce_time = 27.17 / 25 / 2
        yspeed = yacc * bounce_time / 100
        return size, yacc, yspeed, 2

    def pop(self):
        self.kill()
        return (BallLevel1(self.rect.x, self.rect.y, self.display_width, self.display_height, YELLOW, self.fps, False),
                BallLevel1(self.rect.x, self.rect.y, self.display_width, self.display_height, YELLOW, self.fps, True))


class BallLevel3(AbstractBall):
    def load_properties(self, display_width, display_height):
        size = int(display_width / 36.7273)
        yacc = display_height
        bounce_time = 27.17 / 25 / 2
        yspeed = yacc * bounce_time / 100
        return size, yacc, yspeed, 3

    def pop(self):
        self.kill()
        return (BallLevel2(self.rect.x, self.rect.y, self.display_width, self.display_height, BLUE, self.fps, False),
                BallLevel2(self.rect.x, self.rect.y, self.display_width, self.display_height, BLUE, self.fps, True))


class BallLevel4(AbstractBall):
    def load_properties(self, display_width, display_height):
        size = int(display_width / 30.7273)
        yacc = display_height
        bounce_time = 27.17 / 25 / 2
        yspeed = yacc * bounce_time / 100
        return size, yacc, yspeed, 4

    def pop(self):
        self.kill()
        return (BallLevel3(self.rect.x, self.rect.y, self.display_width, self.display_height, GREEN, self.fps, False),
                BallLevel3(self.rect.x, self.rect.y, self.display_width, self.display_height, GREEN, self.fps, True))


class BallLevel5(AbstractBall):
    def load_properties(self, display_width, display_height):
        size = int(display_width / 24.7273)
        yacc = display_height
        bounce_time = 27.17 / 25 / 2
        yspeed = yacc * bounce_time / 100
        return size, yacc, yspeed, 5

    def pop(self):
        self.kill()
        return (BallLevel4(self.rect.x, self.rect.y, self.display_width, self.display_height, RED, self.fps, False),
                BallLevel4(self.rect.x, self.rect.y, self.display_width, self.display_height, RED, self.fps, True))
