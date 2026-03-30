"""Combine multiple demo .npz files into a single file.

Concatenates observations, actions, and level_ids from all input files.
The output filename is auto-generated from the union of covered levels
unless --output is specified.

Usage:
    python src/combine_demos.py checkpoints/user_warmups/demo_lvl_1_2_3_*.npz checkpoints/user_warmups/demo_lvl_9_10_*.npz
    python src/combine_demos.py demos/*.npz --output checkpoints/user_warmups/demo_lvl_all_combined.npz
    python src/combine_demos.py demos/a.npz demos/b.npz --output merged.npz --dry-run
"""

import argparse
import glob
import os
import sys
import time

import numpy as np


def load_demo(path):
    data = np.load(path)
    obs = data["observations"]
    act = data["actions"]
    lvl = data["level_ids"] if "level_ids" in data else np.zeros(len(act), dtype=np.int32)
    rew = data["rewards"] if "rewards" in data else None
    return obs, act, lvl, rew


def main():
    parser = argparse.ArgumentParser(
        description="Combine multiple demo .npz files into one"
    )
    parser.add_argument("files", nargs="+",
                        help="Demo .npz files or glob patterns to combine")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output path (default: auto-named from covered levels)")
    parser.add_argument("--out-dir", type=str, default="checkpoints/user_warmups",
                        help="Directory for auto-named output (default: checkpoints/user_warmups)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be combined without writing")
    args = parser.parse_args()

    # Resolve globs
    paths = []
    for p in args.files:
        expanded = sorted(glob.glob(p))
        paths.extend(expanded if expanded else [p])

    if not paths:
        print("No files found.", file=sys.stderr)
        sys.exit(1)

    # Deduplicate while preserving order
    seen = set()
    paths = [p for p in paths if not (p in seen or seen.add(p))]

    print(f"Combining {len(paths)} file(s):\n")

    all_obs, all_act, all_lvl, all_rew = [], [], [], []
    total_steps = 0
    has_rewards = True

    for path in paths:
        if not os.path.exists(path):
            print(f"  [MISSING] {path}", file=sys.stderr)
            sys.exit(1)
        obs, act, lvl, rew = load_demo(path)
        levels = sorted(np.unique(lvl).tolist())
        rew_tag = f"  rewards={'yes' if rew is not None else 'no'}"
        print(f"  {path}")
        print(f"    {len(act):>7,} steps  levels {levels}{rew_tag}")
        if all_obs and obs.shape[1] != all_obs[0].shape[1]:
            print(f"  [ERROR] Obs dimension mismatch: {path} has {obs.shape[1]} "
                  f"but first file has {all_obs[0].shape[1]}", file=sys.stderr)
            sys.exit(1)
        all_obs.append(obs)
        all_act.append(act)
        all_lvl.append(lvl)
        if rew is not None:
            all_rew.append(rew)
        else:
            has_rewards = False
        total_steps += len(act)

    if not has_rewards and all_rew:
        print("\n  [WARNING] Some files lack rewards — rewards will be omitted from output",
              file=sys.stderr)
        has_rewards = False

    obs_arr = np.concatenate(all_obs, axis=0).astype(np.float32)
    act_arr = np.concatenate(all_act, axis=0).astype(np.int64)
    lvl_arr = np.concatenate(all_lvl, axis=0).astype(np.int32)
    rew_arr = np.concatenate(all_rew, axis=0).astype(np.float32) if has_rewards else None

    covered_levels = sorted(np.unique(lvl_arr).tolist())
    print(f"\nTotal: {total_steps:,} steps  levels {covered_levels}")

    if args.dry_run:
        print("\n[dry-run] No file written.")
        return

    if args.output:
        out_path = args.output
    else:
        os.makedirs(args.out_dir, exist_ok=True)
        lvl_tag = "_".join(str(l) for l in covered_levels)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(args.out_dir, f"demo_lvl_{lvl_tag}_{timestamp}.npz")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    save_kwargs = dict(observations=obs_arr, actions=act_arr, level_ids=lvl_arr)
    if rew_arr is not None:
        save_kwargs["rewards"] = rew_arr
    np.savez_compressed(out_path, **save_kwargs)
    rew_note = "  (with rewards)" if rew_arr is not None else "  (no rewards)"
    print(f"\nSaved -> {out_path}{rew_note}")


if __name__ == "__main__":
    main()
