"""Evaluation script: load a trained model and watch/benchmark it play."""

import argparse
import time
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from bubble_env import BubbleTroubleEnv
from config import NUM_LEVELS


def mask_fn(env):
    """Extract action masks from the environment for MaskablePPO."""
    return env.action_masks()


def evaluate(args):
    """Run evaluation episodes with a trained model."""
    render_mode = None if args.no_render else "human"

    def make_eval_env():
        env = BubbleTroubleEnv(
            render_mode=render_mode,
            enable_powerups=True,
            sequential_levels=True,
            max_level=NUM_LEVELS,
        )
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env

    env = DummyVecEnv([make_eval_env])

    # Load VecNormalize stats if available (reward normalization from training)
    vec_norm_path = args.vec_normalize
    if vec_norm_path:
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False
    else:
        env = VecNormalize(env, norm_obs=False, norm_reward=False)

    model = MaskablePPO.load(args.model)
    print(f"Loaded model from {args.model}")

    total_reward = 0
    total_levels = 0
    total_balls_hit = 0
    total_balls_popped = 0
    games_cleared = 0

    for episode in range(args.episodes):
        obs = env.reset()
        episode_reward = 0
        episode_steps = 0
        ep_levels = 0
        ep_balls_hit = 0
        ep_balls_popped = 0

        while True:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            episode_steps += 1

            info_dict = info[0]
            ep_balls_hit += info_dict.get("balls_hit", 0)
            ep_balls_popped += info_dict.get("balls_popped", 0)
            if info_dict.get("level_cleared"):
                ep_levels += 1
            if info_dict.get("game_cleared"):
                games_cleared += 1

            if render_mode == "human":
                env.render()

            if done[0]:
                break

        total_reward += episode_reward
        total_levels += ep_levels
        total_balls_hit += ep_balls_hit
        total_balls_popped += ep_balls_popped

        status = "CLEARED" if info_dict.get("game_cleared") else ("DIED" if not info_dict.get("TimeLimit.truncated", False) else "TIMEOUT")
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
    eval_parser.add_argument("--vec-normalize", type=str, default=None,
                             help="Path to vecnormalize.pkl from training")

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
