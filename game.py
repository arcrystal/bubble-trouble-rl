import pygame

from player import Player
from ball import Ball
from floor import Floor

import math
import time

class Game:
    """
    Game object for ball breaker.
    """
    WHITE  = (255, 255, 255)
    BLACK  = (  0,   0,   0)
    RED    = (255,   0,   0)
    GREEN  = (  0, 255,   0)
    BLUE   = (  0,   0, 255)
    ORANGE = (255, 255,   0)
    YELLOW = (  0, 255, 255)
    FPS = 60
    UPDATE_MULTIPLIER = 0.1
    DISPLAY_WIDTH = 1280
    DISPLAY_HEIGHT = 960

    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        # Initialize gameplay window
        self.screen = pygame.display.set_mode((Game.DISPLAY_WIDTH, Game.DISPLAY_HEIGHT))
        pygame.display.set_caption("Ball Breaker")
        # Initialize gameplay vars
        self.score = 0
        self.level = 1
        self.font = pygame.font.SysFont('Calibri', 25, True, True)
        self.text = self.font.render(f"Score: {self.score}", True, Game.RED)
        self.screen.blit(self.text, (25, 25))
        self.backgrounds = [
            pygame.image.load("Backgrounds/bluepink.jpg").convert(),
            pygame.image.load("Backgrounds/bluepurple.jpg").convert(),
        ]
        # Initialize sprite Groups
        
        self.objects = [pygame.sprite.Group() for _ in range(10)]
        for group in self.objects:
            group.add(Player())
            group.add(Floor())
        self.objects[0].add(Ball(Game.DISPLAY_WIDTH // 4, Game.DISPLAY_HEIGHT // 6, 0, 0, 0, 9.81, 1))
        self.objects[1].add(Ball(Game.DISPLAY_WIDTH // 4, Game.DISPLAY_HEIGHT // 6, 0, 0, 0, 9.81, 2))
        self.objects[5].add((
            Ball(Game.DISPLAY_WIDTH // 2 - 45, Game.DISPLAY_HEIGHT // 6, 0, 0, 0, 9.81, 0),
            Ball(Game.DISPLAY_WIDTH // 2 - 10, Game.DISPLAY_HEIGHT // 6, 0, 0, 0, 9.81, 1),
            Ball(Game.DISPLAY_WIDTH // 2 + 40, Game.DISPLAY_HEIGHT // 6, 0, 0, 0, 9.81, 2)))

    def play_music(self, filepath):
        pygame.mixer.music.load(filepath)
        pygame.mixer.music.play(-1)

    def load_background(self, background):
        """
        Loads a pygame image as the screen's background.

        Args:
            background (pygame.Surface): raw pygame surface to load.
        Returns:
            Transformed pygame.Surface with the dimensions of the screen.
        Raises:
            None.
        """
        return pygame.transform.scale(background, (Game.DISPLAY_WIDTH, Game.DISPLAY_HEIGHT))

    def test_collision(self, ball, laser):
        x, laserY, laserYstart = laser
        radius = ball.rect.width / 2
        for y in range(int(laserY), int(laserYstart)):
            if radius >= math.dist(ball.rect.center, (x,y)):
                return True

        return False

    def draw_timer(self, timeleft):
        pygame.draw.line(
            self.screen, Game.BLACK,
            (0, Game.DISPLAY_HEIGHT-10),
            (timeleft, Game.DISPLAY_HEIGHT-10),
            10)
        pygame.draw.line(
            self.screen, Game.WHITE,
            (0, Game.DISPLAY_HEIGHT-20),
            (timeleft, Game.DISPLAY_HEIGHT-20),
            10)

    def play(self):
        gameover = False
        nextLevel = False
        laserX = -1
        laserY = -1
        laserYstart = -1
        shooting = False
        i = 0
        clock = pygame.time.Clock()
        for lvl, (lvlsprites, background) in enumerate(zip(self.objects, self.backgrounds)):
            # Check gameover
            if gameover:
                break

            # Set up timers
            timer = 0
            timeleft = Game.DISPLAY_WIDTH
            time.sleep(1)

            # Display new level screen
            curr_background = self.load_background(background)
            self.screen.blit(curr_background, (0, 0))
            lvl_font = self.font.render(f'Level {lvl+1}', True, Game.GREEN, Game.BLUE)
            lvl_font_rect = lvl_font.get_rect()
            lvl_font_rect.center = Game.DISPLAY_WIDTH / 2, Game.DISPLAY_HEIGHT / 10
            self.screen.blit(lvl_font, lvl_font_rect)
            pygame.display.update()
            time.sleep(2)

            # Convert sprite pixels
            player = lvlsprites.sprites()[0]
            for sprites in lvlsprites:
                for key, sprite in sprites.SPRITES.items():
                    sprites.SPRITES[key] = sprite.convert_alpha()
            
            # Play game
            while not (nextLevel or gameover):
                # Get pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        gameover = True

                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_LEFT:
                            player.left()
                        if event.key == pygame.K_RIGHT:
                            player.right()
                        if event.key == pygame.K_UP:
                            if not shooting:
                                laserX, laserY, laserYstart = player.shoot(self.screen)
                                shooting = True
                        if event.key == pygame.K_SPACE:
                            player.jump()
                        
                    if event.type == pygame.KEYUP:
                        if event.key == pygame.K_LEFT and player.xspeed < 0:
                            player.stopx()
                        if event.key == pygame.K_RIGHT and player.xspeed > 0:
                            player.stopx()
                        if event.key == pygame.K_UP:
                            player.stopshoot()

                # Get laser information
                if shooting:
                    laserX, laserY, laserYstart = player.getLaser()
                    if laserX == -1:
                        shooting = False

                # Draw and update screen
                self.screen.blit(curr_background, (0, 0))
                self.draw_timer(timeleft)
                lvlsprites.update(Game.UPDATE_MULTIPLIER)
                lvlsprites.draw(self.screen)
                pygame.display.update()
                clock.tick(Game.FPS)   
                timer += clock.get_time()
                timeleft = Game.DISPLAY_WIDTH - Game.DISPLAY_WIDTH / 20000 * timer        

                if timeleft <= 0:
                    gameover = True
                    print("Time ran out.")
                    break

                for ball in lvlsprites.sprites()[2:]:
                    collide = pygame.sprite.collide_mask(player, ball)
                    if collide:
                        gameover = True
                        print("You lose.")
                        break
                    if laserX != -1:
                        if self.test_collision(ball, (laserX, laserY, laserYstart)):
                            pop_result = ball.pop()
                            if pop_result is None:
                                lvlsprites.remove(ball)
                                # Remove laser from screen
                                player.setLaser(False)
                                shooting = False
                                laserX, laserY, laserYstart = -1, -1, -1
                            else:
                                lvlsprites.add(pop_result)
                                lvlsprites.remove(ball)
                                # Remove laser from screen
                                player.setLaser(False)
                                shooting = False
                                laserX, laserY, laserYstart = -1, -1, -1
                            
                            print(lvlsprites)
                
                if len(lvlsprites) == 2:
                    laserX, laserY, laserYstart = -1, -1, -1
                    nextLevel = True

            nextLevel = False

        pygame.quit()