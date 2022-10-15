import pygame
from numpy import sqrt, arccos

class Ball(pygame.sprite.Sprite):
    """
    A pygame object for the game.
    """
    DISPLAY_WIDTH = 890
    DISPLAY_HEIGHT = DISPLAY_WIDTH / 1.8737
    XSPEED = DISPLAY_WIDTH / 75
    #SPEED = [50, 65, 75, 85, 90, 95, 100]

    # Ball size
    bsizes = [
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
    # [150.855, 311.322, 381.988, 458.35, 530.974]
    bbounce = [
        DISPLAY_HEIGHT * 0.1695,
        DISPLAY_HEIGHT * 0.3498,
        DISPLAY_HEIGHT * 0.4292,
        DISPLAY_HEIGHT * 0.515,
        DISPLAY_HEIGHT * 0.5966,
        DISPLAY_HEIGHT * 0.6803
    ] 
    # values will be multiplied by t=0.1 divided by FPS=52
    SPEED = [sqrt(b*35) for b in bbounce]
    # Multiplied by t = 0.1
    Y_ACC = DISPLAY_WIDTH / 42.657 # 13.91

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
        self.yacceleration = Ball.Y_ACC
        self.ballsize = ballsize
        self.color = color
        self.image = pygame.transform.scale(Ball.SPRITES[color], (Ball.bsizes[ballsize], Ball.bsizes[ballsize]))
        self.rect = self.image.get_rect()
        self.rect.center = (self.x, self.y)

    def bounceX(self):
        if self.x > Ball.DISPLAY_WIDTH / 2:
            self.xspeed = -Ball.XSPEED
        else:
            self.xspeed = Ball.XSPEED

    def bounceY(self):
        self.yspeed = -Ball.SPEED[self.ballsize]

    def update(self, t):
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
        # Update position
        self.x += self.xspeed * t
        self.y += self.yspeed * t
        
        if self.y < 0:
            print("Ceiling pop!")
            # TODO: spawn effect
            self.kill()
            return
        
        # Update speed
        if self.x < 0 or self.x > Ball.DISPLAY_WIDTH - self.rect.width:
            self.xspeed = -self.xspeed

        self.xspeed += self.xacceleration * t
        self.yspeed += self.yacceleration * t

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
        # Velocity of new balls is the sum of the vectors of the ball and laser beam
        if self.ballsize == 0:
            return None
        else:
            newYspeed = -25
            return (Ball(self.x-10, self.y, -Ball.XSPEED, newYspeed, 0, self.ballsize-1, self.color),
                    Ball(self.x+10, self.y, Ball.XSPEED, newYspeed, 0, self.ballsize-1, self.color))

