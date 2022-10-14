import pygame
from numpy import sqrt, arccos

class Ball(pygame.sprite.Sprite):
    """
    A pygame object for the game.
    """
    DISPLAY_WIDTH = 890
    DISPLAY_HEIGHT = DISPLAY_WIDTH / 1.8737
    XSPEED = DISPLAY_WIDTH / 100 # FPS, TIMESTEP
    #SPEED = [50, 65, 75, 85, 90, 95, 100]

    # Ball size
    bsize = [
        DISPLAY_WIDTH / 50.7273, # 55=b1
        DISPLAY_WIDTH / 29.3684, # 95=b2
        DISPLAY_WIDTH / 15.7627, # 177=b3
        DISPLAY_WIDTH / 10.856, # 257=b4
        DISPLAY_WIDTH / 8.2301, # 339=b5
        DISPLAY_WIDTH / 6.7718] # 412=b6

    # Ball bounce height
    b1bounceHeightMax = DISPLAY_WIDTH / 9.1111 / 5.2# 306
    b2bounceHeightMax = DISPLAY_WIDTH / 4.5663 / 5.2 # 611
    b3bounceHeightMax = DISPLAY_WIDTH / 3.4317 / 5.2 # 813
    b4bounceHeightMax = DISPLAY_WIDTH / 2.7542 / 5.2 # 1013
    b5bounceHeightMax = DISPLAY_WIDTH / 2.2928 / 5.2 # 1217
    b6bounceHeightMax = DISPLAY_WIDTH / 1
    SPEED = [b1bounceHeightMax,b2bounceHeightMax,b3bounceHeightMax,b4bounceHeightMax,b5bounceHeightMax]

    SPRITES = {
        0:pygame.image.load("Sprites/ball1_sm.png"),
        1:pygame.image.load("Sprites/ball2_sm.png"),
        2:pygame.image.load("Sprites/ball3_sm.png"),
        3:pygame.image.load("Sprites/ball4_sm.png"),
        4:pygame.image.load("Sprites/ball5_sm.png"),
        5:pygame.image.load("Sprites/ball6_sm.png"),
        6:pygame.image.load("Sprites/ball7_sm.png")}

    for i in range(6):
        SPRITES[i] = pygame.transform.scale(SPRITES[i], (bsize[i], bsize[i]))

    Y_ACC = DISPLAY_WIDTH / 64

    def __init__(self, x, y, xspeed, yspeed, xacceleration, ballsize):
        assert ballsize < 5
        super().__init__() # equivalent to pygame.sprite.Sprite.__init__(self)
        self.x = x
        self.y = y
        self.xspeed = xspeed
        self.yspeed = yspeed
        self.xacceleration = xacceleration
        self.yacceleration = Ball.Y_ACC
        self.ballsize = ballsize
        self.image = Ball.SPRITES[ballsize]
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
            newYspeed = -5
            return (Ball(self.x-10, self.y, -Ball.XSPEED, newYspeed, 0, self.ballsize-1),
                    Ball(self.x+10, self.y, Ball.XSPEED, newYspeed, 0, self.ballsize-1))
            

