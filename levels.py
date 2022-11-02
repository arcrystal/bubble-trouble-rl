from ball import Ball
from game import DISPLAY_WIDTH, DISPLAY_HEIGHT
b1bounce = DISPLAY_WIDTH / 10

# Ball(x, y, xspeed, yspeed, xacceleration, ballsize, color)
class Levels:
    def __init__(self):
        pass

    def get(self, lvl):
        if lvl == 1:
            return [Ball(DISPLAY_WIDTH // 5, DISPLAY_HEIGHT // 4, 'right', 0, 0, 1, 'yellow')]
        elif lvl == 2:
            return [Ball(DISPLAY_WIDTH // 4, DISPLAY_HEIGHT // 4, 0, 0, 0, 2, 'green')]
        elif lvl == 3:
            return [Ball(DISPLAY_WIDTH // 4, DISPLAY_HEIGHT // 4, 0, 0, 0, 3, 'red')]
        elif lvl == 4:
            return [Ball(DISPLAY_WIDTH // 4, DISPLAY_HEIGHT // 4, 'left', 0, 0, 2, 'orange'),
                    Ball(3 * DISPLAY_WIDTH // 4, DISPLAY_HEIGHT // 4, 'right', 0, 0, 2, 'orange')]
        elif lvl == 5:
            return [Ball(DISPLAY_WIDTH // 3, DISPLAY_HEIGHT // 4, 0, 0, 0, 2, 'yellow'),
                    Ball(2*DISPLAY_WIDTH // 3 - 10, DISPLAY_HEIGHT // 4, 0, 0, 0, 3, 'green')]
        elif lvl == 6:
            return [Ball(  DISPLAY_WIDTH // 7, 394, b1bounce, 0, 0, 0, 'purple'),
                    Ball(2*DISPLAY_WIDTH // 7, 394, b1bounce, 0, 0, 0, 'purple'),
                    Ball(3*DISPLAY_WIDTH // 7, 394, b1bounce, 0, 0, 0, 'purple'),
                    Ball(4*DISPLAY_WIDTH // 7, 394, b1bounce, 0, 0, 0, 'purple'),
                    Ball(5*DISPLAY_WIDTH // 7, 394, b1bounce, 0, 0, 0, 'purple'),
                    Ball(6*DISPLAY_WIDTH // 7, 394, b1bounce, 0, 0, 0, 'purple')]
        elif lvl == 7:
            return [Ball(  DISPLAY_WIDTH // 7 - 40, 394,  b1bounce, 0, 0, 0, 'red'),
                    Ball(  DISPLAY_WIDTH // 7 - 20, 394,  b1bounce, 0, 0, 0, 'orange'),
                    Ball(  DISPLAY_WIDTH // 7,      394,  b1bounce, 0, 0, 0, 'yellow'),
                    Ball(2*DISPLAY_WIDTH // 7 - 40, 394,  b1bounce, 0, 0, 0, 'red'),
                    Ball(2*DISPLAY_WIDTH // 7 - 20, 394,  b1bounce, 0, 0, 0, 'orange'),
                    Ball(2*DISPLAY_WIDTH // 7     , 394,  b1bounce, 0, 0, 0, 'red'),
                    Ball(5*DISPLAY_WIDTH // 7 + 10,  394, -b1bounce, 0, 0, 0, 'red'),
                    Ball(5*DISPLAY_WIDTH // 7 + 30, 394, -b1bounce, 0, 0, 0, 'orange'),
                    Ball(5*DISPLAY_WIDTH // 7 + 50, 394, -b1bounce, 0, 0, 0, 'red'),
                    Ball(6*DISPLAY_WIDTH // 7 + 10    , 394, -b1bounce, 0, 0, 0, 'yellow'),
                    Ball(6*DISPLAY_WIDTH // 7 + 30, 394, -b1bounce, 0, 0, 0, 'orange'),
                    Ball(6*DISPLAY_WIDTH // 7 + 50, 394, -b1bounce, 0, 0, 0, 'red')]
        elif lvl == 8:
            return [Ball(DISPLAY_WIDTH // 2, DISPLAY_HEIGHT // 4, 0, 0, 0, 4, 'red')]
        else:
            raise ValueError("Invalid level.")
        