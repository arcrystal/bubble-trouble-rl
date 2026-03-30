"""Human-playable Bubble Trouble using the headless engine + pygame renderer.

Controls:
    Left/Right arrows — move
    Space — shoot
    R — restart level
    Q/Escape — quit

Usage:
    python src/play.py              # regular mode, level 1
    python src/play.py 5            # regular mode, start from level 5
    python src/play.py --infinity   # infinity mode, random seed
    python src/play.py --infinity 42  # infinity mode, fixed seed
"""

import argparse
import sys
import numpy as np
import pygame
from bubble_env import BubbleTroubleEnv
from config import NUM_LEVELS


def main():
    parser = argparse.ArgumentParser(description="Play Bubble Trouble")
    parser.add_argument("level", nargs="?", type=int, default=1,
                        help="Start level (regular mode only, default: 1)")
    parser.add_argument("--infinity", nargs="?", type=int, const=None, default=False,
                        metavar="SEED",
                        help="Play infinity mode (optional: fixed seed)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (e.g. 2 for 2x, 5 for 5x)")
    args = parser.parse_args()

    if args.infinity is not False:
        from infinity_env import InfinityModeEnv
        env = InfinityModeEnv(seed=args.infinity, render_mode="human",
                              render_speed=args.speed)
        mode_label = f"Infinity (seed={args.infinity})" if args.infinity is not None else "Infinity"
    else:
        start_level = max(1, min(args.level, NUM_LEVELS))
        env = BubbleTroubleEnv(
            render_mode="human",
            enable_powerups=True,
            sequential_levels=True,
            start_level=start_level,
            max_level=NUM_LEVELS,
            render_speed=args.speed,
        )
        mode_label = f"Level {start_level}"

    print(f"Playing: {mode_label}")
    obs, info = env.reset()
    env.render()

    running = True
    shoot_this_frame = False
    while running:
        shoot_this_frame = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                elif event.key == pygame.K_SPACE:
                    shoot_this_frame = True

        # MultiDiscrete: [move, shoot]
        # Move: 0=LEFT, 1=RIGHT, 2=STILL
        # Shoot: 0=SHOOT, 1=NO_SHOOT
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            move = 0  # LEFT
        elif keys[pygame.K_RIGHT]:
            move = 1  # RIGHT
        else:
            move = 2  # STILL

        shoot = 0 if shoot_this_frame else 1  # SHOOT / NO_SHOOT
        action = np.array([move, shoot])

        if not running:
            break

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            if args.infinity is not False:
                elapsed = info.get("elapsed_time", 0.0)
                mins = int(elapsed) // 60
                secs = int(elapsed) % 60
                print(f"Survived {mins}:{secs:02d} "
                      f"(difficulty: {info.get('difficulty', 0):.0%})")
            elif info.get("game_cleared"):
                print("You cleared all levels!")
            elif terminated:
                print(f"Game over on level {info.get('current_level', '?')}. "
                      f"Levels cleared: {info.get('levels_cleared', 0)}")
            else:
                print(f"Time out on level {info.get('current_level', '?')}.")

            # Wait a moment, then restart
            pygame.time.wait(1500)
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()
