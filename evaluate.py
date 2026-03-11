"""Evaluation script: load a trained model and watch/benchmark it play."""

import argparse
import time
import numpy as np
from stable_baselines3 import PPO
from bubble_env import BubbleTroubleEnv
from config import NUM_LEVELS


def evaluate(args):
    """Run evaluation episodes with a trained model."""
    render_mode = None if args.no_render else "human"

    env = BubbleTroubleEnv(
        render_mode=render_mode,
        enable_powerups=True,
        sequential_levels=True,
        max_level=NUM_LEVELS,
    )

    model = PPO.load(args.model)
    print(f"Loaded model from {args.model}")

    total_reward = 0
    total_levels = 0
    total_balls_hit = 0
    total_balls_popped = 0
    games_cleared = 0

    for episode in range(args.episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        ep_levels = 0
        ep_balls_hit = 0
        ep_balls_popped = 0

        while True:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1

            ep_balls_hit += info.get("balls_hit", 0)
            ep_balls_popped += info.get("balls_popped", 0)
            if info.get("level_cleared"):
                ep_levels += 1
            if info.get("game_cleared"):
                games_cleared += 1

            if render_mode == "human":
                env.render()

            if terminated or truncated:
                break

        total_reward += episode_reward
        total_levels += ep_levels
        total_balls_hit += ep_balls_hit
        total_balls_popped += ep_balls_popped

        status = "CLEARED" if info.get("game_cleared") else ("DIED" if terminated else "TIMEOUT")
        print(f"Episode {episode + 1}/{args.episodes}: "
              f"reward={episode_reward:.2f}, steps={episode_steps}, "
              f"levels={ep_levels}, balls_hit={ep_balls_hit}, "
              f"balls_popped={ep_balls_popped}, status={status}")

    n = args.episodes
    print(f"\n--- Summary ({n} episodes) ---")
    print(f"Avg reward:       {total_reward / n:.2f}")
    print(f"Avg levels:       {total_levels / n:.2f}")
    print(f"Avg balls hit:    {total_balls_hit / n:.2f}")
    print(f"Avg balls popped: {total_balls_popped / n:.2f}")
    print(f"Games cleared:    {games_cleared}/{n}")

    env.close()


def benchmark(args):
    """Benchmark raw environment steps per second (no model, random actions)."""
    env = BubbleTroubleEnv(
        render_mode=None,
        enable_powerups=True,
        sequential_levels=False,
    )

    n_steps = args.benchmark_steps
    obs, _ = env.reset()
    start = time.perf_counter()

    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    elapsed = time.perf_counter() - start
    sps = n_steps / elapsed
    print(f"Benchmark: {n_steps:,} steps in {elapsed:.2f}s = {sps:,.0f} steps/sec")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate or benchmark Bubble Trouble RL")
    subparsers = parser.add_subparsers(dest="command")

    # Evaluate subcommand
    eval_parser = subparsers.add_parser("play", help="Watch trained agent play")
    eval_parser.add_argument("model", type=str, help="Path to saved model (.zip)")
    eval_parser.add_argument("--episodes", type=int, default=10)
    eval_parser.add_argument("--no-render", action="store_true")
    eval_parser.add_argument("--deterministic", action="store_true", default=True)

    # Benchmark subcommand
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark env speed")
    bench_parser.add_argument("--benchmark-steps", type=int, default=100_000)

    args = parser.parse_args()

    if args.command == "play":
        evaluate(args)
    elif args.command == "benchmark":
        benchmark(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
