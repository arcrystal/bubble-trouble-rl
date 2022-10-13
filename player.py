
import pygame

class Player(pygame.sprite.Sprite):
    """
    Player object for ball breaker Game().
    """
    RED = (255, 0, 0)
    DISPLAY_WIDTH = 1280
    DISPLAY_HEIGHT = 960
    JUMP_CONSTANT = 8
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

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def setY(self, y):
        self.y = y

    def setX(self, x):
        self.x = x
    
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

