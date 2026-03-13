"""Bubble Trouble RL — MaskablePPO training.

Two-phase strategy (default):
  Phase 1 — Warmup:    curriculum learning across 12 levels (learns basic skills)
  Phase 2 — Per-level: one dedicated env per level, all levels in parallel (fine-tunes every level)

Single-phase flags:
  --warmup-only           run only Phase 1 (curriculum)
  --skip-warmup --resume  run only Phase 2 from a saved checkpoint

Usage:
  python src/train.py                                   # full two-phase run
  python src/train.py --warmup-steps 150000000          # longer warmup
  python src/train.py --warmup-only                     # curriculum only
  python src/train.py --skip-warmup --resume checkpoints/warmup_model.zip
  python src/train.py --timesteps 1000000 --n-envs 4    # quick smoke test
"""

import argparse
import os
import sys
import torch

# When running as a script, src/ is automatically on sys.path.
# Nothing extra needed — all sibling modules (config, engine, etc.) resolve directly.

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import (
    SubprocVecEnv, DummyVecEnv, VecNormalize, VecFrameStack
)
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.monitor import Monitor

from bubble_env import BubbleTroubleEnv
from config import TRAINING, CURRICULUM, NUM_LEVELS


# ---------------------------------------------------------------------------
# Action mask helper
# ---------------------------------------------------------------------------

def mask_fn(env):
    return env.action_masks()


# ---------------------------------------------------------------------------
# Environment factories
# ---------------------------------------------------------------------------

def make_env(rank, enable_powerups=False, max_level=2, sequential=True, seed=None):
    """Standard env for curriculum training."""
    def _init():
        env = BubbleTroubleEnv(
            enable_powerups=enable_powerups,
            max_level=max_level,
            sequential_levels=sequential,
        )
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        env.reset(seed=(seed + rank) if seed is not None else None)
        return env
    return _init


def make_env_pinned(level, enable_powerups=False, seed=None, rank=0):
    """Env pinned to a single level — resets to that level every episode."""
    def _init():
        env = BubbleTroubleEnv(
            enable_powerups=enable_powerups,
            start_level=level,
            max_level=level,
            sequential_levels=True,
        )
        env = ActionMasker(env, mask_fn)
        env = Monitor(env)
        env.reset(seed=(seed + rank) if seed is not None else None)
        return env
    return _init


# ---------------------------------------------------------------------------
# Vectorized env builders
# ---------------------------------------------------------------------------

def _wrap(venv, norm_reward, n_stack):
    venv = VecNormalize(venv, norm_obs=False, norm_reward=norm_reward, clip_reward=10.0)
    venv = VecFrameStack(venv, n_stack=n_stack)
    return venv


def build_curriculum_env(n_envs, max_lvl, powerups, n_stack, seed):
    fns = [make_env(i, enable_powerups=powerups, max_level=max_lvl, seed=seed)
           for i in range(n_envs)]
    venv = SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns)
    return _wrap(venv, norm_reward=True, n_stack=n_stack)


def build_perlevel_env(envs_per_level, n_stack, seed):
    env_fns = [
        make_env_pinned(level, enable_powerups=True, seed=seed,
                        rank=(level - 1) * envs_per_level + k)
        for level in range(1, NUM_LEVELS + 1)
        for k in range(envs_per_level)
    ]
    n = len(env_fns)
    venv = SubprocVecEnv(env_fns) if n > 1 else DummyVecEnv(env_fns)
    return _wrap(venv, norm_reward=True, n_stack=n_stack)


def build_eval_env(n_stack):
    base = DummyVecEnv([make_env(0, enable_powerups=True, max_level=NUM_LEVELS,
                                 sequential=True, seed=42)])
    return _wrap(base, norm_reward=False, n_stack=n_stack)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class CurriculumCallback(BaseCallback):
    """Shifts env difficulty through CURRICULUM phases."""

    def __init__(self, curriculum, verbose=0):
        super().__init__(verbose)
        self.curriculum = sorted(curriculum, key=lambda x: x[0])
        self.current_phase = 0

    def _on_step(self):
        new_phase = self.current_phase
        for i, (start_ts, *_) in enumerate(self.curriculum):
            if self.num_timesteps >= start_ts:
                new_phase = i

        if new_phase != self.current_phase:
            self.current_phase = new_phase
            _, min_lvl, max_lvl, powerups = self.curriculum[new_phase]
            if self.verbose:
                print(f"[Curriculum] Phase {new_phase}: levels {min_lvl}-{max_lvl}, "
                      f"powerups={'ON' if powerups else 'OFF'} @ step {self.num_timesteps}")
            env = self.training_env
            while hasattr(env, 'venv'):
                env = env.venv
            if hasattr(env, 'env_method'):
                env.env_method('set_curriculum',
                               start_level=min_lvl, max_level=max_lvl,
                               enable_powerups=powerups)
        return True


class MetricsCallback(BaseCallback):
    """Logs per-episode game stats to TensorBoard."""

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep = info["episode"]
                self.logger.record("game/episode_reward", ep["r"])
                self.logger.record("game/episode_length", ep["l"])
                self.logger.record("game/levels_cleared", info.get("levels_cleared", 0))
                self.logger.record("game/highest_level", info.get("highest_level", 1))
                self.logger.record("game/game_cleared", 1 if info.get("game_cleared") else 0)
                self.logger.record("game/shots_wasted", info.get("shots_wasted", 0))
                self.logger.record("game/danger_splits", info.get("danger_splits", 0))
        return True


class EntropyScheduleCallback(BaseCallback):
    """Linearly decays ent_coef — MaskablePPO doesn't call callable ent_coef in train()."""

    def __init__(self, ent_start, ent_end, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.ent_start = ent_start
        self.ent_end = ent_end
        self.total_timesteps = total_timesteps

    def _on_step(self):
        p = max(0.0, 1.0 - self.num_timesteps / self.total_timesteps)
        self.model.ent_coef = float(self.ent_end + (self.ent_start - self.ent_end) * p)
        return True


def linear_schedule(start, end, total_timesteps):
    def schedule(progress_remaining):
        return end + (start - end) * progress_remaining
    return schedule


def make_callbacks(checkpoint_dir, log_dir, eval_env, n_envs,
                   total_timesteps, with_curriculum=False):
    cb = [
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(checkpoint_dir, "best"),
            log_path=os.path.join(log_dir, "eval"),
            eval_freq=max(50_000 // n_envs, 500),
            n_eval_episodes=10, deterministic=True, verbose=1,
        ),
        CheckpointCallback(
            save_freq=max(10_000_000 // n_envs, 1000),
            save_path=os.path.join(checkpoint_dir, "periodic"),
            name_prefix="ppo_bubble", verbose=1,
        ),
        MetricsCallback(),
        EntropyScheduleCallback(
            TRAINING["ent_coef_start"], TRAINING["ent_coef_end"], total_timesteps
        ),
    ]
    if with_curriculum:
        cb.insert(2, CurriculumCallback(CURRICULUM, verbose=1))
    return CallbackList(cb)


# ---------------------------------------------------------------------------
# Model constructor
# ---------------------------------------------------------------------------

def build_model(env, lr_schedule, device, seed, log_dir):
    return MaskablePPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=lr_schedule,
        n_steps=TRAINING["n_steps"],
        batch_size=TRAINING["batch_size"],
        n_epochs=TRAINING["n_epochs"],
        gamma=TRAINING["gamma"],
        gae_lambda=TRAINING["gae_lambda"],
        clip_range=TRAINING["clip_range"],
        ent_coef=TRAINING["ent_coef_start"],
        vf_coef=TRAINING["vf_coef"],
        max_grad_norm=TRAINING["max_grad_norm"],
        policy_kwargs=dict(
            net_arch=dict(pi=TRAINING["net_arch_pi"], vf=TRAINING["net_arch_vf"]),
            activation_fn=torch.nn.ReLU,
        ),
        tensorboard_log=log_dir,
        verbose=1,
        device=device,
        seed=seed,
    )


def save(model, env, checkpoint_dir, name):
    path = os.path.join(checkpoint_dir, name)
    model.save(path)
    vecnorm = env
    while not isinstance(vecnorm, VecNormalize):
        vecnorm = vecnorm.venv
    vecnorm.save(os.path.join(checkpoint_dir, f"{name}_vecnorm.pkl"))
    return path


# ---------------------------------------------------------------------------
# Training phases
# ---------------------------------------------------------------------------

def run_warmup(args, device, n_stack, total_timesteps, lr_schedule):
    """Phase 1: curriculum warmup. Returns (checkpoint_path, timesteps_done)."""
    _, min_lvl, max_lvl, powerups = CURRICULUM[0]
    n_envs = args.n_envs or TRAINING["n_envs"]

    print(f"\n{'='*60}")
    print(f"Phase 1 — Warmup  ({args.warmup_steps:,} steps, curriculum)")
    print(f"  {n_envs} envs  |  initial levels: {min_lvl}–{max_lvl}")
    print(f"{'='*60}\n")

    env = build_curriculum_env(n_envs, max_lvl, powerups, n_stack, args.seed)
    eval_env = build_eval_env(n_stack)
    model = build_model(env, lr_schedule, device, args.seed, args.log_dir)

    model.learn(
        total_timesteps=args.warmup_steps,
        callback=make_callbacks(
            checkpoint_dir=os.path.join(args.checkpoint_dir, "warmup"),
            log_dir=args.log_dir, eval_env=eval_env,
            n_envs=n_envs, total_timesteps=total_timesteps,
            with_curriculum=True,
        ),
        progress_bar=True,
        reset_num_timesteps=True,
    )

    path = save(model, env, args.checkpoint_dir, "warmup_model")
    ts = model.num_timesteps
    print(f"\nWarmup complete — saved to {path}.zip  ({ts:,} steps)")

    env.close()
    eval_env.close()
    return path, ts


def run_perlevel(args, device, n_stack, total_timesteps, lr_schedule,
                 resume_path, start_timestep):
    """Phase 2: per-level fine-tuning loaded from resume_path."""
    envs_per_level = args.envs_per_level
    n_envs = envs_per_level * NUM_LEVELS
    remaining = total_timesteps - start_timestep

    print(f"\n{'='*60}")
    print(f"Phase 2 — Per-level  ({remaining:,} steps)")
    print(f"  {n_envs} envs  ({envs_per_level} × {NUM_LEVELS} levels, all in parallel)")
    print(f"{'='*60}\n")

    env = build_perlevel_env(envs_per_level, n_stack, args.seed)
    eval_env = build_eval_env(n_stack)

    model = MaskablePPO.load(
        resume_path, env=env, device=device,
        custom_objects={"learning_rate": lr_schedule,
                        "n_steps": TRAINING["n_steps"]},
    )
    model.num_timesteps = start_timestep
    model._num_timesteps_at_start = start_timestep

    model.learn(
        total_timesteps=remaining,
        callback=make_callbacks(
            checkpoint_dir=os.path.join(args.checkpoint_dir, "perlevel"),
            log_dir=args.log_dir, eval_env=eval_env,
            n_envs=n_envs, total_timesteps=total_timesteps,
            with_curriculum=False,
        ),
        progress_bar=True,
        reset_num_timesteps=False,
    )

    save(model, env, args.checkpoint_dir, "final_model")
    print(f"\nPer-level phase complete.")

    env.close()
    eval_env.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train Bubble Trouble RL (warmup → per-level by default)"
    )
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Total training timesteps (default: from config)")
    parser.add_argument("--warmup-steps", type=int, default=100_000_000,
                        help="Steps for warmup phase (default: 100M). "
                             "Set to 0 with --skip-warmup to go straight to per-level.")
    parser.add_argument("--warmup-only", action="store_true",
                        help="Run only the curriculum warmup phase")
    parser.add_argument("--skip-warmup", action="store_true",
                        help="Skip warmup and go straight to per-level (requires --resume)")
    parser.add_argument("--envs-per-level", type=int, default=2,
                        help="Envs per level in per-level phase (default: 2 → 24 total envs)")
    parser.add_argument("--n-envs", type=int, default=None,
                        help="Envs for warmup phase (default: from config)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Warmup checkpoint path for --skip-warmup")
    parser.add_argument("--resume-timestep", type=int, default=None,
                        help="Timestep at which --resume was saved (default: --warmup-steps)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    total_timesteps = args.timesteps or TRAINING["total_timesteps"]
    n_stack = TRAINING.get("n_frame_stack", 4)
    lr_schedule = linear_schedule(
        TRAINING["learning_rate_start"], TRAINING["learning_rate_end"], total_timesteps
    )

    print(f"Total timesteps: {total_timesteps:,}")
    print(f"LR: {TRAINING['learning_rate_start']} → {TRAINING['learning_rate_end']}")
    print(f"Entropy: {TRAINING['ent_coef_start']} → {TRAINING['ent_coef_end']}")
    print(f"Frame stack: {n_stack}, gamma: {TRAINING['gamma']}, "
          f"gae_lambda: {TRAINING['gae_lambda']}")
    print(f"Network: pi={TRAINING['net_arch_pi']}, vf={TRAINING['net_arch_vf']}")

    if args.warmup_only:
        # Curriculum only — no per-level phase
        run_warmup(args, device, n_stack, total_timesteps, lr_schedule)

    elif args.skip_warmup or args.resume:
        # Per-level only, from a saved checkpoint
        if not args.resume:
            parser.error("--skip-warmup requires --resume <checkpoint>")
        start_ts = args.resume_timestep if args.resume_timestep is not None else args.warmup_steps
        run_perlevel(args, device, n_stack, total_timesteps, lr_schedule,
                     resume_path=args.resume, start_timestep=start_ts)

    else:
        # Default: warmup then per-level
        warmup_path, warmup_ts = run_warmup(
            args, device, n_stack, total_timesteps, lr_schedule
        )
        run_perlevel(args, device, n_stack, total_timesteps, lr_schedule,
                     resume_path=warmup_path, start_timestep=warmup_ts)


if __name__ == "__main__":
    main()
