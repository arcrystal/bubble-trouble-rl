
class Utils:

    def __init__(self):
        pass

    def getCirclePixels(self, centerx, centery, radius):
        for x, y in product(range(int(radius) + 1), repeat=2):
            if x**2 + y**2 <= radius**2:
                yield from set((
                    (centerx + x, centery + y),
                    (centerx + x, centery-y),
                    (centerx-x, centery+y),
                    (centerx-x, centery-y),))

 