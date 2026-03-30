"""Simulate many infinity mode games and plot max balls on screen per episode.

Usage:
  python src/analyze_ball_pressure.py \
      checkpoints/infinity/best/best_model.zip \
      --vec-normalize checkpoints/infinity/best/best_model_vecnorm.pkl \
      --episodes 200
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from infinity_env import InfinityModeEnv


def mask_fn(env):
    return env.action_masks()


def run(args):
    def make_env():
        env = InfinityModeEnv(seed=None)
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        return env

    venv = DummyVecEnv([make_env])
    if args.vec_normalize:
        venv = VecNormalize.load(args.vec_normalize, venv)
        venv.training = False
        venv.norm_reward = False
    else:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=False)

    model = MaskablePPO.load(args.model, device="cpu")
    print(f"Loaded {args.model}")
    print(f"Simulating {args.episodes} episodes (headless)...\n")

    max_balls_per_episode = []

    for ep in range(args.episodes):
        obs = venv.reset()
        ep_max_balls = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, _, dones, infos = venv.step(action)
            done = dones[0]

            # Read n_balls directly from the inner env
            inner = venv.venv.envs[0]
            while hasattr(inner, "env"):
                inner = inner.env
            n_balls = inner.engine.n_balls
            if n_balls > ep_max_balls:
                ep_max_balls = n_balls

        max_balls_per_episode.append(ep_max_balls)
        elapsed = infos[0].get("elapsed_time", 0.0)
        print(f"  ep {ep+1:>4}/{args.episodes}  max_balls={ep_max_balls:>3}  "
              f"survived={int(elapsed)//60}:{int(elapsed)%60:02d}")

    venv.close()

    arr = np.array(max_balls_per_episode)
    print(f"\n--- Max balls on screen ({args.episodes} episodes) ---")
    print(f"  mean:   {arr.mean():.1f}")
    print(f"  median: {np.median(arr):.1f}")
    print(f"  min:    {arr.min()}")
    print(f"  max:    {arr.max()}")
    print(f"  p25:    {np.percentile(arr, 25):.1f}")
    print(f"  p75:    {np.percentile(arr, 75):.1f}")

    bins = range(0, arr.max() + 2)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(arr, bins=bins, align="left", rwidth=0.8, color="steelblue", edgecolor="white")
    ax.set_xlabel("Max balls on screen in episode", fontsize=12)
    ax.set_ylabel("Number of episodes", fontsize=12)
    ax.set_title(f"Infinity mode — max ball pressure per episode (n={args.episodes})", fontsize=13)
    ax.set_xticks(range(0, arr.max() + 1, max(1, arr.max() // 20)))
    ax.axvline(arr.mean(), color="tomato", linestyle="--", linewidth=1.5, label=f"mean={arr.mean():.1f}")
    ax.axvline(np.median(arr), color="gold", linestyle="--", linewidth=1.5, label=f"median={np.median(arr):.1f}")
    ax.legend()
    plt.tight_layout()

    out = args.output
    plt.savefig(out, dpi=150)
    print(f"\nHistogram saved to {out}")
    if not args.no_show:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to model .zip")
    parser.add_argument("--vec-normalize", default=None, help="Path to vecnorm .pkl")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--output", default="ball_pressure_histogram.png")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
