from game import Game


def main():
    game = Game({"render_mode": "human", "fps":48})
    game.play()


if __name__ == "__main__":
    main()
