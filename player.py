
import pygame

class Player(pygame.sprite.Sprite):
    """
    Player object for ball breaker Game().
    """
    RED = (255, 0, 0)
    DISPLAY_WIDTH = 1280
    DISPLAY_HEIGHT = 960
    JUMP_CONSTANT = 8
    LASER_SPEED = DISPLAY_HEIGHT / 100
    SPEED = DISPLAY_WIDTH / 50
    SPRITES = {
        "still":pygame.image.load("Sprites/person_still_sm.png"),
        "shoot":pygame.image.load("Sprites/person_shoot_sm.png"),
        "left":pygame.image.load("Sprites/person_left_sm.png"),
        "right":pygame.image.load("Sprites/person_right_sm.png")}
    
    def __init__(self):
        super().__init__() # equivalent to pygame.sprite.Sprite.__init__(self)
        self.xspeed = 0
        self.yspeed = 0
        self.yacceleration = 0
        self.image = Player.SPRITES['still']
        self.rect = self.image.get_rect()
        self.x = Player.DISPLAY_WIDTH / 2
        self.y = 9 * Player.DISPLAY_HEIGHT / 10 - self.rect.height
        self.rect.x = self.x
        self.rect.y = self.y
        # Laser Variables
        self.laser = False
        self.laserX = -1
        self.laserYstart = -1
        self.laserY = -1
        self.screen = None

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def setY(self, y):
        self.y = y

    def setX(self, x):
        self.x = x

    def getLaser(self):
        return self.laserX, self.laserY, self.laserYstart

    def setLaser(self, laserStatus):
        self.laser = laserStatus
        if not laserStatus:
            self.laserX = -1
            self.laserY = -1
            self.laserYstart = -1

    def stopshoot(self):
        print("Stop shooting.")
        self.image = Player.SPRITES['still']
        self.rect = self.image.get_rect()


    def shoot(self, screen):
        print("Shoot.")
        if self.laser:
            return self.getLaser()

        self.laser = True
        self.laserX = self.getX() + self.rect.width / 2
        self.laserYstart = self.getY()
        self.laserY = self.getY()
        self.image = Player.SPRITES['shoot']
        self.rect = self.image.get_rect()
        self.screen = screen
        return self.getLaser()
    
    def stopx(self):
        print("Stop moving.")
        self.xspeed = 0
        self.image = Player.SPRITES['still']
        self.rect = self.image.get_rect()

    def stopy(self):
        self.yspeed = 0
        self.yacceleration = 0

    def left(self):
        print("Left.")
        self.xspeed = -Player.SPEED
        self.image = Player.SPRITES['left']
        self.rect = self.image.get_rect()

    def right(self):
        print("Right.")
        self.xspeed = Player.SPEED
        self.image = Player.SPRITES['right']
        self.rect = self.image.get_rect()

    def jump(self):
        if self.y < 9 * Player.DISPLAY_HEIGHT / 10:
            return
        print("Jump.")
        self.yspeed = -(Player.SPEED * Player.JUMP_CONSTANT)
        self.yacceleration = 9.81

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
        if self.y > 9 * Player.DISPLAY_HEIGHT / 10:
            self.y = 9 * Player.DISPLAY_HEIGHT / 10
            self.stopy()

        self.x += self.xspeed * t
        self.y += self.yspeed * t
        self.yspeed += self.yacceleration

        self.rect.x = max(0, min(self.x, Player.DISPLAY_WIDTH))
        self.rect.y = self.y

        if self.laser:
            self.laserY -= Player.LASER_SPEED
            if self.laserY < 0:
                self.laser = False
                self.laserX = self.laserY = -1
                self.screen = None
            else:
                pygame.draw.line(self.screen, Player.RED, (self.laserX, self.laserYstart), (self.laserX, self.laserY))


        