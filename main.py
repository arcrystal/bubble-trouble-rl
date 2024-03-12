import numpy
from game import Game
import asyncio

game = Game({"render_mode": "human", "fps": 48})
async def main():
    while(True):
        game.reset()
        await game.play()

asyncio.run(main())
