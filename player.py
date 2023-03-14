import pygame
from game import TIMESTEP, DISPLAY_WIDTH, DISPLAY_HEIGHT


sprites = {
        "still":pygame.image.load("Sprites/person_still_sm.png"),
        "shoot":pygame.image.load("Sprites/person_shoot_sm.png"),
        "left":pygame.image.load("Sprites/person_left_sm.png"),
        "right":pygame.image.load("Sprites/person_right_sm.png")}

resize = DISPLAY_WIDTH / 890
for key, val in sprites.items():
    sprites[key] = pygame.transform.scale(val, (val.get_rect().width * resize, val.get_rect().height * resize))


class Player(pygame.sprite.Sprite):
    """
    Player object for ball breaker Game().
    """
    RED = (255, 0, 0)
    SPEED = DISPLAY_WIDTH / 5
    SPRITES = sprites
    
    def __init__(self):
        super().__init__() # equivalent to pygame.sprite.Sprite.__init__(self)
        self.x = DISPLAY_WIDTH / 2
        self.y = DISPLAY_HEIGHT
        self.xspeed = 0
        self.yspeed = 0
        self.yacceleration = 0
        self.image = Player.SPRITES['still']
        self.rect = self.image.get_rect()
        self.y -= self.image.get_height()
        self.rect.x = self.x
        self.rect.y = self.y
        self.width = self.rect.width

    def getWidth(self):
        return self.width

    def getX(self):
        return self.rect.centerx

    def getY(self):
        return self.y

    def setY(self, y):
        self.y = y

    def setX(self, x):
        self.x = x
    
    def stop(self):
        # print("Stop moving.")
        self.xspeed = 0
        self.image = Player.SPRITES['still']
        self.rect = self.image.get_rect()
        self.rect.x = self.x
        self.rect.y = self.y

    def left(self):
        # print("Left.")
        self.xspeed = -Player.SPEED
        self.image = Player.SPRITES['left']
        self.rect = self.image.get_rect()
        self.rect.x = self.x
        self.rect.y = self.y
        return 'left'

    def right(self):
        # print("Right.")
        self.xspeed = Player.SPEED
        self.image = Player.SPRITES['right']
        self.rect = self.image.get_rect()
        self.rect.x = self.x
        self.rect.y = self.y
        return 'right'

    def bad_move(self, direction):
        if (self.x > DISPLAY_WIDTH - self.width and 
            self.xspeed > 0 and 
            direction=='right'):
            return True
        if (self.x < 0 and
            self.xspeed < 0 and
            direction=='left'):
            return True
        
        return False

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
        if self.x > DISPLAY_WIDTH - self.width and self.xspeed > 0:
            self.x = DISPLAY_WIDTH - self.width
            self.rect.x = self.x
            self.stop()
            return
        elif self.x < 0 and self.xspeed < 0:
            self.x = 0
            self.rect.x = self.x
            self.stop()
            return

        self.x += self.xspeed * TIMESTEP
        self.rect.x = self.x
