import pygame
from game import TIMESTEP, DISPLAY_WIDTH, DISPLAY_HEIGHT

YACC = [
    # DISPLAY HEIGHT --> ballY == 229
    DISPLAY_HEIGHT, # DONE
    DISPLAY_HEIGHT * 1.03, # DONE
    DISPLAY_HEIGHT * 0.82, # DONE
    DISPLAY_HEIGHT * 0.95, # DONE
    DISPLAY_HEIGHT * 1, # NOT
    DISPLAY_HEIGHT * 1 # NOT
]

class Ball(pygame.sprite.Sprite):
    """
    A pygame object for the game.
    """
    XSPEED = DISPLAY_WIDTH / 5
    # Ball size
    sizes = [
        DISPLAY_WIDTH / 50.7273, # 55=b1
        DISPLAY_WIDTH / 29.3684, # 95=b2
        DISPLAY_WIDTH / 15.7627, # 177=b3
        DISPLAY_WIDTH / 10.856, # 257=b4
        DISPLAY_WIDTH / 8.2301, # 339=b5
        DISPLAY_WIDTH / 6.7718] # 412=b6

    SPRITES = {
        'yellow':pygame.image.load("Sprites/ball_yellow.png"),
        'red':pygame.image.load("Sprites/ball_red.png"),
        'blue':pygame.image.load("Sprites/ball_blue.png"),
        'purple':pygame.image.load("Sprites/ball_purple.png"),
        'green':pygame.image.load("Sprites/ball_green.png"),
        'orange':pygame.image.load("Sprites/ball_orange.png"),
        'pink':pygame.image.load("Sprites/ball_pink.png")}

    # Ball bounce height (floor to bottom of ball)
    
    BOUNCE_HEIGHT = [
        DISPLAY_HEIGHT - DISPLAY_HEIGHT * 0.1695,
        DISPLAY_HEIGHT - DISPLAY_HEIGHT * 0.3498, # this num is 166.1536 right now ball bounces 312
        DISPLAY_HEIGHT - DISPLAY_HEIGHT * 0.4292, # this num is 203.868 but right now ball bounces 238
        DISPLAY_HEIGHT - DISPLAY_HEIGHT * 0.515,
        DISPLAY_HEIGHT - DISPLAY_HEIGHT * 0.5966,
        DISPLAY_HEIGHT - DISPLAY_HEIGHT * 0.6803
    ]
    bounce_time = [
        # seconds / bounce / 2
        27.17 / 25 / 2,
        35.29 / 23 / 2,
        32.08 / 18 / 2,
        17.48 / 10 / 2,
        -1,
        -1
    ]
    YSPEED = [YACC[i] * t for i, t in enumerate(bounce_time)]

    def __init__(self, x, y, xspeed, yspeed, xacceleration, ballsize, color):
        assert ballsize < 5
        super().__init__() # equivalent to pygame.sprite.Sprite.__init__(self)
        self.x = x
        self.y = y
        if type(xspeed) in [float, int]:
            self.xspeed = xspeed
        elif xspeed == 'left':
            self.xspeed = -Ball.XSPEED
        elif xspeed == 'right':
            self.xspeed = Ball.XSPEED
        else:
            self.xspeed = 0

        self.yspeed = yspeed
        self.xacceleration = xacceleration
        self.ballsize = ballsize
        self.color = color
        self.image = pygame.transform.scale(
            Ball.SPRITES[color],
            (Ball.sizes[ballsize], Ball.sizes[ballsize]))
        self.rect = self.image.get_rect()
        self.rect.center = (self.x, self.y)

    def bounceX(self):
        if self.x > DISPLAY_WIDTH / 2:
            self.xspeed = -Ball.XSPEED
        else:
            self.xspeed = Ball.XSPEED

    def bounceY(self):
        self.yspeed = -Ball.YSPEED[self.ballsize]

    def update(self):
        """
        Overides pygame.sprite.Sprite.update()
        Applied when Group.update() is called.

        Args:
            t (float): the timestep of the update.
        Returns:
            None.
        Raises:
            None.
        """
        # if self.yspeed < 10 and self.yspeed > -10:
        #     print("H", round(Ball.BOUNCE_HEIGHT[self.ballsize]), " :  ballY", round(self.y))
        # Update position
        self.x += self.xspeed * TIMESTEP
        # y = y0 + v0yt + ½at2
        y0 = self.y
        self.y += self.yspeed * TIMESTEP + 0.5 * YACC[self.ballsize] * TIMESTEP ** 2
        
        if self.y < 0:
            print("Ceiling pop!")
            self.kill()
            return
        
        # Update speed
        if self.x < 0 or self.x > DISPLAY_WIDTH - self.rect.width:
            self.xspeed = -self.xspeed

        self.xspeed += self.xacceleration * TIMESTEP
        # vy2	 = v0y2 − 2g(y − y0)
        self.yspeed += YACC[self.ballsize] * TIMESTEP

        # Update rect dimensions
        self.rect.x = self.x
        self.rect.y = self.y

    def pop(self):
        """
        Occurs when the player hits the ball with their laser.

        Args:
            None.
        Returns:
            2 balls of size n-1.
        Raises:
            None.
        """
        print("Laser Pop!")
        if self.ballsize == 0:
            return
        else:
            newYspeed = -25
            return (Ball(self.x-10, self.y, -Ball.XSPEED, newYspeed,
                    0, self.ballsize-1, self.color),
                    Ball(self.x+10, self.y, Ball.XSPEED, newYspeed,
                    0, self.ballsize-1, self.color))