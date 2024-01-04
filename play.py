from game import Game


def main():
    while(True):
        game = Game({"render_mode": "human", "fps":200})
    #game.balls = game.levels.get(6)
        game.play()


if __name__ == "__main__":
    main()
