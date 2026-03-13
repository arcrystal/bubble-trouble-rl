"""Training script for Bubble Trouble RL using sb3-contrib MaskablePPO."""

import argparse
import os
import torch
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize, VecFrameStack
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from bubble_env import BubbleTroubleEnv
from config import TRAINING, CURRICULUM, NUM_LEVELS


def mask_fn(env):
    """Extract action masks from the environment for MaskablePPO."""
    return env.action_masks()


def make_env(rank, enable_powerups=False, max_level=2, sequential=True, seed=None):
    """Create a factory function for a monitored BubbleTroubleEnv."""
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


def linear_schedule(start, end, total_timesteps):
    """Return a function that linearly interpolates from start to end."""
    def schedule(progress_remaining):
        # progress_remaining goes from 1.0 → 0.0
        return end + (start - end) * progress_remaining
    return schedule


class CurriculumCallback(BaseCallback):
    """Adjusts environment difficulty based on training progress."""

    def __init__(self, curriculum, verbose=0):
        super().__init__(verbose)
        self.curriculum = sorted(curriculum, key=lambda x: x[0])
        self.current_phase = 0

    def _on_step(self):
        timestep = self.num_timesteps
        # Find the current phase
        new_phase = self.current_phase
        for i, (start_ts, min_lvl, max_lvl, powerups) in enumerate(self.curriculum):
            if timestep >= start_ts:
                new_phase = i

        if new_phase != self.current_phase:
            self.current_phase = new_phase
            _, min_lvl, max_lvl, powerups = self.curriculum[new_phase]
            if self.verbose:
                print(f"[Curriculum] Phase {new_phase}: levels {min_lvl}-{max_lvl}, "
                      f"powerups={'ON' if powerups else 'OFF'} at step {timestep}")

            # Update all sub-environments
            # With VecNormalize + VecFrameStack wrapping, we need to get to the base env
            env = self.training_env
            # Unwrap to find the env that has env_method
            while hasattr(env, 'venv'):
                env = env.venv
            if hasattr(env, 'env_method'):
                env.env_method(
                    'set_curriculum',
                    start_level=min_lvl,
                    max_level=max_lvl,
                    enable_powerups=powerups,
                )
        return True


class MetricsCallback(BaseCallback):
    """Logs custom game metrics to TensorBoard."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._episode_rewards = []

    def _on_step(self):
        # Log per-episode info from Monitor wrapper
        # SB3's Monitor adds "episode" key to info on the terminal step only
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep = info["episode"]
                self.logger.record("game/episode_reward", ep["r"])
                self.logger.record("game/episode_length", ep["l"])
                # Episode-level cumulative stats (available on every step, but only log at episode end)
                self.logger.record("game/levels_cleared", info.get("levels_cleared", 0))
                self.logger.record("game/highest_level", info.get("highest_level", 1))
                self.logger.record("game/game_cleared", 1 if info.get("game_cleared") else 0)
                self.logger.record("game/shots_wasted", info.get("shots_wasted", 0))
                self.logger.record("game/danger_splits", info.get("danger_splits", 0))
        return True


def train(args):
    """Run MaskablePPO training."""
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Determine initial curriculum phase
    phase_idx = args.curriculum_phase
    if phase_idx >= len(CURRICULUM):
        phase_idx = len(CURRICULUM) - 1
    _, min_lvl, max_lvl, powerups = CURRICULUM[phase_idx]

    # Create vectorized environments
    n_envs = args.n_envs or TRAINING["n_envs"]
    if n_envs > 1:
        env = SubprocVecEnv([
            make_env(i, enable_powerups=powerups, max_level=max_lvl, seed=args.seed)
            for i in range(n_envs)
        ])
    else:
        env = DummyVecEnv([
            make_env(0, enable_powerups=powerups, max_level=max_lvl, seed=args.seed)
        ])

    # VecNormalize: normalize rewards only (obs already in [-1, 1])
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # Frame stacking for temporal context (ball trajectories)
    n_stack = TRAINING.get("n_frame_stack", 4)
    env = VecFrameStack(env, n_stack=n_stack)

    # Create eval environment (same wrappers but separate normalization stats)
    eval_base = DummyVecEnv([
        make_env(0, enable_powerups=True, max_level=NUM_LEVELS,
                 sequential=True, seed=42)
    ])
    eval_env = VecNormalize(eval_base, norm_obs=False, norm_reward=False)
    eval_env = VecFrameStack(eval_env, n_stack=n_stack)

    # Determine device
    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    print(f"Using device: {device}")

    total_timesteps = args.timesteps or TRAINING["total_timesteps"]

    # Build schedules for learning rate and entropy coefficient
    lr_schedule = linear_schedule(
        TRAINING["learning_rate_start"], TRAINING["learning_rate_end"], total_timesteps
    )
    ent_schedule = linear_schedule(
        TRAINING["ent_coef_start"], TRAINING["ent_coef_end"], total_timesteps
    )

    # Build or load model
    if args.resume:
        print(f"Resuming from {args.resume}")
        model = MaskablePPO.load(args.resume, env=env, device=device)
    else:
        model = MaskablePPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=lr_schedule,
            n_steps=TRAINING["n_steps"],
            batch_size=TRAINING["batch_size"],
            n_epochs=TRAINING["n_epochs"],
            gamma=TRAINING["gamma"],
            gae_lambda=TRAINING["gae_lambda"],
            clip_range=TRAINING["clip_range"],
            ent_coef=ent_schedule,
            vf_coef=TRAINING["vf_coef"],
            max_grad_norm=TRAINING["max_grad_norm"],
            policy_kwargs=dict(
                net_arch=dict(
                    pi=TRAINING["net_arch_pi"],
                    vf=TRAINING["net_arch_vf"],
                ),
                activation_fn=torch.nn.ReLU,
            ),
            tensorboard_log=args.log_dir,
            verbose=1,
            device=device,
            seed=args.seed,
        )

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.checkpoint_dir, "best"),
        log_path=os.path.join(args.log_dir, "eval"),
        eval_freq=max(50_000 // n_envs, 500),  # ~every 50k total steps
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(10_000_000 // n_envs, 1000),  # ~every 10M total steps
        save_path=os.path.join(args.checkpoint_dir, "periodic"),
        name_prefix="ppo_bubble",
        verbose=1,
    )

    curriculum_callback = CurriculumCallback(CURRICULUM, verbose=1)
    metrics_callback = MetricsCallback(verbose=0)

    callbacks = CallbackList([
        eval_callback, checkpoint_callback, curriculum_callback, metrics_callback
    ])

    # Train
    print(f"Training for {total_timesteps:,} timesteps with {n_envs} envs")
    print(f"  LR: {TRAINING['learning_rate_start']} → {TRAINING['learning_rate_end']}")
    print(f"  Entropy: {TRAINING['ent_coef_start']} → {TRAINING['ent_coef_end']}")
    print(f"  Frame stack: {n_stack}, gamma: {TRAINING['gamma']}, gae_lambda: {TRAINING['gae_lambda']}")
    print(f"  Network: pi={TRAINING['net_arch_pi']}, vf={TRAINING['net_arch_vf']}")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model + VecNormalize stats
    final_path = os.path.join(args.checkpoint_dir, "final_model")
    model.save(final_path)
    # Save normalization stats (needed for evaluation)
    vec_norm = env
    while not isinstance(vec_norm, VecNormalize):
        vec_norm = vec_norm.venv
    vec_norm.save(os.path.join(args.checkpoint_dir, "vecnormalize.pkl"))
    print(f"Final model saved to {final_path}")

    env.close()
    eval_env.close()


def main():
    parser = argparse.ArgumentParser(description="Train Bubble Trouble RL agent")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Total training timesteps (default: from config)")
    parser.add_argument("--n-envs", type=int, default=None,
                        help="Number of parallel environments (default: from config)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "mps", "cuda"],
                        help="PyTorch device")
    parser.add_argument("--log-dir", type=str, default="./logs",
                        help="TensorBoard log directory")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Checkpoint save directory")
    parser.add_argument("--curriculum-phase", type=int, default=0,
                        help="Starting curriculum phase (0-indexed)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
