import pygame
import os

from player import Player
from ball import Ball
from floor import Floor
from laser import Laser

FPS = float(os.environ.get('FPS'))
TIMESTEP = 1 / FPS
DISPLAY_WIDTH = float(os.environ.get('DISPLAY_WIDTH'))
DISPLAY_HEIGHT = DISPLAY_WIDTH / 1.8737

class Game:
    """
    Game object for falling circles.
    """
    WHITE  = (255, 255, 255)
    BLACK  = (  0,   0,   0)
    RED    = (255,   0,   0)
    GREEN  = (  0, 255,   0)
    BLUE   = (  0,   0, 255)
    ORANGE = (255, 255,   0)
    YELLOW = (  0, 255, 255)
    LVL_TIME = [20000, 35000, 50000, 65000, 80000, 90000]

    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        # Initialize gameplay window
        self.screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT+27+10)) # + platform_h + timer_h
        pygame.display.set_caption("Ball Breaker")
        # Initialize gameplay vars
        self.score = 0
        self.level = 1
        self.font = pygame.font.SysFont('Calibri', 25, True, True)
        self.text = self.font.render(f"Score: {self.score}", True, Game.RED)
        self.screen.blit(self.text, (25, 25))
        self.backgrounds = [pygame.image.load("Backgrounds/bluepink.jpg").convert()] * len(Game.LVL_TIME)

    def init_level(self, lvl, lives=None, newlvl=True):
        """
        Returns:
            lvlsprites (pygame.sprite.Group): contains all sprites
            player: (pygame.sprite.Sprite): player sprite
            balls (pygame.sprite.Group): contains all ball sprites
            platform: (pygame.sprite.Sprite): platform sprite
            background: (pygame.Surface): background from game screen
            timer (float): keeps track of time elapsed
            timeleft (float): keeps track of time with respect to display size
        """
        balls = pygame.sprite.Group()
        if lvl == 1:
            balls.add(Ball(DISPLAY_WIDTH // 4, DISPLAY_HEIGHT // 6, 0, 0, 0, 1, 'yellow'))
        elif lvl == 2:
            balls.add(Ball(DISPLAY_WIDTH // 4, DISPLAY_HEIGHT // 6, 0, 0, 0, 2, 'green'))
        elif lvl == 3:
            balls.add(Ball(DISPLAY_WIDTH // 4, DISPLAY_HEIGHT // 6, 0, 0, 0, 3, 'red'))
        elif lvl == 4:
            balls.add(
                Ball(DISPLAY_WIDTH // 4, DISPLAY_HEIGHT // 6, 'left', 0, 0, 2, 'orange'),
                Ball(3 * DISPLAY_WIDTH // 4, DISPLAY_HEIGHT // 6, 'right', 0, 0, 2, 'orange'))
        elif lvl == 5:
            balls.add(
                Ball(DISPLAY_WIDTH // 3, DISPLAY_HEIGHT // 6, 0, 0, 0, 2, 'yellow'),
                Ball(2*DISPLAY_WIDTH // 3 - 10, DISPLAY_HEIGHT // 6, 0, 0, 0, 3, 'green'))

        player = Player()
        platform = Floor()

        for sprites in balls:
            for color, sprite in sprites.SPRITES.items():
                sprites.SPRITES[color] = sprite.convert_alpha()

        for key, sprite in player.SPRITES.items():
                player.SPRITES[key] = sprite.convert_alpha()

        platform.image = platform.image.convert_alpha()

        lvlsprites = pygame.sprite.Group()
        lvlsprites.add(player)
        lvlsprites.add(balls)
        lvlsprites.add(platform)

        if newlvl:
            lvl_font = self.font.render(f'Level {lvl}', True, Game.GREEN, Game.BLUE)
            lvl_font_rect = lvl_font.get_rect()
            lvl_font_rect.center = DISPLAY_WIDTH / 2, DISPLAY_HEIGHT / 10
        else:
            lives_font = self.font.render(f'Lives: {lives}', True, Game.RED)
            lives_font_rect = lives_font.get_rect()
            lives_font_rect.center = DISPLAY_WIDTH / 2, DISPLAY_HEIGHT / 10
            self.screen.blit(lives_font, lives_font_rect)

        pygame.display.update()
        pygame.time.wait(500)
        return lvlsprites, player, balls, platform, self.backgrounds[lvl], 0, DISPLAY_WIDTH

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
        return pygame.transform.scale(background, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    def draw_timer(self, timeleft):
        pygame.draw.line(
            self.screen, Game.BLACK,
            (0, DISPLAY_HEIGHT+27),
            (timeleft, DISPLAY_HEIGHT+27),
            10)

    def collide(self, laser, ball):
        if laser.rect.x < ball.x + ball.image.get_width() and laser.rect.x + laser.image.get_width() < ball.rect.x:
            if laser.rect.y < ball.rect.y + ball.image.get_height():
                return True

        return False

    def play(self):
        gameover = False
        nextLevel = False
        shooting = False
        laser = None
        clock = pygame.time.Clock()
        lives = 5
        for lvl in range(1, 6):
            if gameover:
                break

            lvlsprites, player, balls, platform, background, timer, timeleft = self.init_level(lvl)
            while not (nextLevel or gameover):
                # Get pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        exit()

                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_LEFT:
                            player.left()
                        if event.key == pygame.K_RIGHT:
                            player.right()
                        if event.key == pygame.K_UP and not shooting:
                            shooting = True
                            laser = Laser(player.rect.centerx)
                        if event.key == pygame.K_i:
                            pygame.image.save(self.screen, "screenshot.png")

                    if event.type == pygame.KEYUP:
                        if event.key == pygame.K_LEFT and player.xspeed < 0:
                            player.stop()
                        if event.key == pygame.K_RIGHT and player.xspeed > 0:
                            player.stop()

                # Draw and update screen
                self.screen.blit(background, (0, 0))

                # Get collision updates
                for ball in balls:
                    if pygame.sprite.collide_mask(player, ball):
                        lives -= 1
                        print("Lost life.", lives, "left.")
                        if lives == 0:
                            print("You lose.")
                            gameover = True
                            break

                        shooting = False
                        lvlsprites, player, balls, platform, background, timer, timeleft \
                            = self.init_level(lvl, lives, False)
                        break

                    if shooting:
                        laser.update()
                        if pygame.sprite.collide_mask(laser, ball):
                            print("Laser pop.")
                            shooting = False
                            pop_result = ball.pop()
                            lvlsprites.remove(ball)
                            balls.remove(ball)
                            if pop_result is not None:
                                lvlsprites.add(pop_result)
                                balls.add(pop_result)
                                shooting = False
                        elif laser.hitCeiling():
                            shooting = False
                        else:
                            self.screen.blit(laser.curr, laser.rect)

                    if pygame.sprite.collide_rect(ball, platform):
                        ball.bounceY()

                self.draw_timer(timeleft)
                lvlsprites.update()
                lvlsprites.draw(self.screen)
                pygame.display.update()
                # means that for every (1 / FPS) seconds, FPS frames should pass.
                clock.tick(FPS)
                timer += clock.get_time()
                timeleft = DISPLAY_WIDTH - DISPLAY_WIDTH / Game.LVL_TIME[lvl-1] * timer
                if timeleft <= 0:
                    lives -= 1
                    print("Time ran out.")
                    print("Lost life.", lives, "left.")
                    if lives == 0:
                        print("You lose.")
                        gameover = True
                        break

                    shooting = False
                    lvlsprites, player, balls, platform, background, timer, timeleft \
                        = self.init_level(lvl, lives, False)
                elif not balls:
                    nextLevel = True

            nextLevel = False

        pygame.quit()