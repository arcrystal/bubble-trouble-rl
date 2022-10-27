from ball import Ball
from game import DISPLAY_WIDTH, DISPLAY_HEIGHT
b1bounce = DISPLAY_WIDTH / 10

LEVELS = {
    1: Ball(DISPLAY_WIDTH // 4, DISPLAY_HEIGHT // 6, 0, 0, 0, 1, 'yellow'),
    2: Ball(DISPLAY_WIDTH // 4, DISPLAY_HEIGHT // 6, 0, 0, 0, 2, 'green') ,
    3: Ball(DISPLAY_WIDTH // 4, DISPLAY_HEIGHT // 6, 0, 0, 0, 3, 'red')   ,
    4: (Ball(DISPLAY_WIDTH // 4, DISPLAY_HEIGHT // 6, 'left', 0, 0, 2, 'orange'),
        Ball(3 * DISPLAY_WIDTH // 4, DISPLAY_HEIGHT // 6, 'right', 0, 0, 2, 'orange')),
    5: (Ball(DISPLAY_WIDTH // 3, DISPLAY_HEIGHT // 6, 0, 0, 0, 2, 'yellow'),
        Ball(2*DISPLAY_WIDTH // 3 - 10, DISPLAY_HEIGHT // 6, 0, 0, 0, 3, 'green')),
    6: (Ball(  DISPLAY_WIDTH // 7, 394, b1bounce, 0, 0, 0, 'purple'),
        Ball(2*DISPLAY_WIDTH // 7, 394, b1bounce, 0, 0, 0, 'purple'),
        Ball(3*DISPLAY_WIDTH // 7, 394, b1bounce, 0, 0, 0, 'purple'),
        Ball(4*DISPLAY_WIDTH // 7, 394, b1bounce, 0, 0, 0, 'purple'),
        Ball(5*DISPLAY_WIDTH // 7, 394, b1bounce, 0, 0, 0, 'purple'),
        Ball(6*DISPLAY_WIDTH // 7, 394, b1bounce, 0, 0, 0, 'purple')),
    7: (Ball(  DISPLAY_WIDTH // 7 - 40, 394,  b1bounce, 0, 0, 0, 'red'),
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
        Ball(6*DISPLAY_WIDTH // 7 + 50, 394, -b1bounce, 0, 0, 0, 'red')),
    8: Ball(DISPLAY_WIDTH // 2, DISPLAY_HEIGHT // 6, 0, 0, 0, 4, 'red')
}