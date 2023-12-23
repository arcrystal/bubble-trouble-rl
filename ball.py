
from AbstractBall import AbstractBall

RED = (255, 0, 0)
YELLOW = (245, 237, 7)
GREEN = (52, 145, 33)
BLUE = (128, 206, 242)
ORANGE = (237, 141, 45)

def calculate_vertical_motion(height, time_between_peaks):
    """
    Calculate the vertical acceleration and maximum vertical speed of a ball
    bouncing with no loss of speed in a game environment.

    Parameters:
    height (float): The height of the bounce in meters.
    time_between_peaks (float): The time between the peaks of the bounce in seconds.

    Returns:
    tuple: A tuple containing the vertical acceleration (in m/s^2) and
           the maximum vertical speed (in m/s).
    """
    time_to_peak = time_between_peaks / 2
    acceleration = 2 * height / time_to_peak**2
    max_speed = acceleration * time_to_peak
    return acceleration, max_speed

# Example usage
height_example = 5  # height in meters
time_between_peaks_example = 2  # time between peaks in seconds
calculate_vertical_motion(height_example, time_between_peaks_example)


class BallLevel1(AbstractBall):
    def load_properties(self, display_width):
        radius = round(13 / 720 * display_width / 2)
        bounce_height = round(66 / 720 * display_width)
        bounce_time = 1.2667
        yacc, max_speed = calculate_vertical_motion(bounce_height, bounce_time)
        return radius, max_speed, yacc

    def pop(self):
        self.kill()


class BallLevel2(AbstractBall):
    def load_properties(self, display_width):
        radius = round(26 / 720 * display_width / 2)
        bounce_height = round(133 / 720 * display_width)
        bounce_time = 1.6667
        yacc, max_speed = calculate_vertical_motion(bounce_height, bounce_time)
        return radius, max_speed, yacc

    def pop(self):
        self.kill()
        return (BallLevel1(self.rect.x, self.rect.y, self.display_width, self.display_height, YELLOW, self.fps, False),
                BallLevel1(self.rect.x, self.rect.y, self.display_width, self.display_height, YELLOW, self.fps, True))


class BallLevel3(AbstractBall):
    def load_properties(self, display_width):
        radius = round(46 / 720 * display_width / 2)
        bounce_height = round(165 / 720 * display_width)
        bounce_time = 1.9333
        yacc, max_speed = calculate_vertical_motion(bounce_height, bounce_time)
        return radius, max_speed, yacc

    def pop(self):
        self.kill()
        return (BallLevel2(self.rect.x, self.rect.y, self.display_width, self.display_height, BLUE, self.fps, False),
                BallLevel2(self.rect.x, self.rect.y, self.display_width, self.display_height, BLUE, self.fps, True))


class BallLevel4(AbstractBall):
    def load_properties(self, display_width):
        radius = round(67 / 720 * display_width / 2)
        bounce_height = round(195 / 720 * display_width)
        bounce_time = -1
        yacc, max_speed = calculate_vertical_motion(bounce_height, bounce_time)
        return radius, max_speed, yacc

    def pop(self):
        self.kill()
        return (BallLevel3(self.rect.x, self.rect.y, self.display_width, self.display_height, GREEN, self.fps, False),
                BallLevel3(self.rect.x, self.rect.y, self.display_width, self.display_height, GREEN, self.fps, True))
