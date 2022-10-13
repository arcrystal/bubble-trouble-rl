import pygame
from numpy import sqrt, arccos

class Ball(pygame.sprite.Sprite):
    """
    A pygame object for the game.
    """
    DISPLAY_WIDTH = 1280
    DISPLAY_HEIGHT = 960
    SPEED = [50, 80, 130, 210, 340, 340, 340]
    SPRITES = {
        0:pygame.image.load("Sprites/ball1_sm.png"),
        1:pygame.image.load("Sprites/ball2_sm.png"),
        2:pygame.image.load("Sprites/ball3_sm.png"),
        3:pygame.image.load("Sprites/ball4_sm.png"),
        4:pygame.image.load("Sprites/ball5_sm.png"),
        5:pygame.image.load("Sprites/ball6_sm.png"),
        6:pygame.image.load("Sprites/ball7_sm.png")}

    def __init__(self, x, y, xspeed, yspeed, xacceleration, yacceleration, ballsize):
        assert ballsize < 5
        super().__init__() # equivalent to pygame.sprite.Sprite.__init__(self)
        self.x = x
        self.y = y
        self.xspeed = xspeed
        self.yspeed = yspeed
        self.xacceleration = xacceleration
        self.yacceleration = yacceleration
        self.ballsize = ballsize
        self.image = Ball.SPRITES[ballsize]
        self.rect = self.image.get_rect()
        self.rect.center = (self.x, self.y)

    def bounceX(self):
        self.xspeed = -self.xspeed

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
            newYspeed = self.yspeed - 25 if self.yspeed < 0 else -(5 / self.yspeed) * 25
            return (Ball(self.x-10, self.y, -(10-self.ballsize)*2, newYspeed, 0, 9.81, self.ballsize-1),
                    Ball(self.x+10, self.y, (10-self.ballsize)*2, newYspeed, 0, 9.81, self.ballsize-1))
            

