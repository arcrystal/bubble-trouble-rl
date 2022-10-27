from game import Game
import argparse
 


def main():
    # Initialize parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-u", "--User",
        help="Flag to play pygame as user",
        action="store_true")
    args = parser.parse_args()
    
    # Play game with specified params
    Game().play(user=args.User)

if __name__=="__main__":
    main()