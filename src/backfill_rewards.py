"""Backfill rewards into existing demo .npz files that lack them.

Replays each demo's actions through the environment to reconstruct the
reward signal. The observations in the file are NOT replaced — only a
'rewards' array is added.

This works because the env is deterministic: given the same level and
action sequence, the same rewards are produced. The script resets the
env to each level before replaying that level's actions.

Usage:
    python src/backfill_rewards.py checkpoints/user_warmups/demo_lvl_*.npz
    python src/backfill_rewards.py checkpoints/user_warmups/demo_lvl_all.npz --dry-run
"""

import argparse
import glob
import os
import sys

import numpy as np

from bubble_env import BubbleTroubleEnv


def backfill_file(path, dry_run=False):
    """Add rewards to a single demo .npz file by replaying actions."""
    data = np.load(path)
    if "rewards" in data:
        print(f"  {path}: already has rewards — skipping")
        return True

    obs = data["observations"]
    actions = data["actions"]
    level_ids = data["level_ids"] if "level_ids" in data else np.ones(len(actions), dtype=np.int32)

    levels = np.unique(level_ids)
    total_steps = len(actions)
    rewards = np.zeros(total_steps, dtype=np.float32)

    print(f"  {path}: {total_steps:,} steps, levels {levels.tolist()}")

    env = BubbleTroubleEnv(
        start_level=int(levels[0]),
        max_level=int(levels[-1]),
        enable_powerups=False,
        sequential_levels=True,
    )

    # Split by contiguous runs (not unique levels) so combined demos with
    # the same level recorded twice are handled correctly.
    boundaries = np.where(np.diff(level_ids) != 0)[0] + 1
    runs = np.split(np.arange(total_steps), boundaries)

    replay_ok = True
    for run_indices in runs:
        level = int(level_ids[run_indices[0]])
        run_actions = actions[run_indices]

        env.set_curriculum(
            start_level=level,
            max_level=level,
            enable_powerups=False,
        )
        env.reset()

        for i, idx in enumerate(run_indices):
            action = run_actions[i]
            _, reward, terminated, truncated, info = env.step(action)
            rewards[idx] = reward

            if (terminated or truncated) and i < len(run_indices) - 1:
                # Episode ended before all actions replayed — env state diverged.
                remaining = len(run_indices) - i - 1
                print(f"    WARNING: Level {level} terminated early ({remaining} actions remaining)")
                replay_ok = False
                break

    env.close()

    if not replay_ok:
        print(f"    Replay diverged — rewards may be partial")

    n_nonzero = np.count_nonzero(rewards)
    print(f"    Rewards: {n_nonzero:,} non-zero of {total_steps:,} "
          f"(range [{rewards.min():.3f}, {rewards.max():.3f}])")

    if dry_run:
        print(f"    [dry-run] Would overwrite {path}")
        return replay_ok

    # Save back with rewards added
    save_kwargs = dict(
        observations=obs,
        actions=actions,
        level_ids=level_ids,
        rewards=rewards,
    )
    np.savez_compressed(path, **save_kwargs)
    print(f"    Saved (with rewards) -> {path}")
    return replay_ok


def main():
    parser = argparse.ArgumentParser(
        description="Backfill rewards into demo .npz files by replaying actions"
    )
    parser.add_argument("files", nargs="+",
                        help="Demo .npz files or glob patterns")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without writing")
    args = parser.parse_args()

    paths = []
    for p in args.files:
        expanded = sorted(glob.glob(p))
        paths.extend(expanded if expanded else [p])

    if not paths:
        print("No files found.", file=sys.stderr)
        sys.exit(1)

    # Deduplicate
    seen = set()
    paths = [p for p in paths if not (p in seen or seen.add(p))]

    print(f"Backfilling rewards for {len(paths)} file(s):\n")

    all_ok = True
    for path in paths:
        if not os.path.exists(path):
            print(f"  [MISSING] {path}", file=sys.stderr)
            all_ok = False
            continue
        ok = backfill_file(path, dry_run=args.dry_run)
        if not ok:
            all_ok = False
        print()

    if all_ok:
        print("Done — all files processed successfully.")
    else:
        print("Done — some files had warnings (see above).")


if __name__ == "__main__":
    main()
