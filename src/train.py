"""Bubble Trouble RL — MaskablePPO two-phase training.

Phase 1 (Warmup): Progressive curriculum across levels 1-12.
  All envs see the same level range that advances with training.
  Agent learns fundamental skills: move, shoot, dodge, pop.

Phase 2 (Level-Distributed Mastery): All levels trained simultaneously.
  One env per level (round-robin). Every PPO batch contains transitions
  from all levels. The `current_level` obs feature lets the policy
  specialize per level. No catastrophic forgetting — every level is
  always in the training distribution.

Why PPO over DQN/QR-DQN for this task:
  - Full trajectory rollouts → better credit assignment for ceiling pop chains
  - Entropy-regularized exploration → structured discovery of novel strategies
  - Graceful curriculum shifts (no stale replay buffer)
  - No frame stacking: peak_height + velocity obs features encode temporal context

Usage:
  python src/train.py                                      # full two-phase (200M steps)
  python src/train.py --warmup-steps 100000000             # longer warmup
  python src/train.py --warmup-only                        # curriculum only
  python src/train.py --skip-warmup --resume checkpoints/warmup_model.zip
  python src/train.py --timesteps 1000000 --n-envs 4       # quick smoke test
"""

import argparse
import os
import sys

import numpy as np
import torch as th

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.monitor import Monitor

from bubble_env import BubbleTroubleEnv
from config import TRAINING as T, CURRICULUM, NUM_LEVELS


# ---------------------------------------------------------------------------
# Environment factories
# ---------------------------------------------------------------------------

def _mask_fn(env):
    """Extract action masks from BubbleTroubleEnv for MaskablePPO."""
    return env.action_masks()


def make_env(rank: int, start_level: int = 1, max_level: int = NUM_LEVELS,
             enable_powerups: bool = False, seed: int | None = None):
    def _init():
        env = BubbleTroubleEnv(
            start_level=start_level,
            max_level=max_level,
            enable_powerups=enable_powerups,
            sequential_levels=True,
        )
        env = ActionMasker(env, _mask_fn)
        env = Monitor(env)
        env.reset(seed=(seed + rank) if seed is not None else None)
        return env
    return _init


def build_env(n_envs: int, start_level: int, max_level: int,
              powerups: bool, seed: int | None) -> VecNormalize:
    """All workers see the same level range (for curriculum warmup)."""
    fns = [make_env(i, start_level, max_level, powerups, seed) for i in range(n_envs)]
    venv = SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns)
    return VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=50.0)


def build_level_distributed_env(n_envs: int, num_levels: int,
                                seed: int | None) -> VecNormalize:
    """Each worker pinned to a specific level (round-robin).

    With n_envs >= num_levels, every level gets at least one dedicated worker.
    With fewer envs, levels are spread evenly across the range.
    """
    fns = []
    for i in range(n_envs):
        if n_envs >= num_levels:
            level = (i % num_levels) + 1
        else:
            # Spread evenly: e.g. 4 envs across 22 levels → levels 1, 8, 15, 22
            level = int(1 + i * (num_levels - 1) / max(1, n_envs - 1))
        fns.append(make_env(i, start_level=level, max_level=level,
                            enable_powerups=True, seed=seed))
    venv = SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns)
    return VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=50.0)


def build_eval_env() -> VecNormalize:
    """Eval env: full sequential game with raw (unnormalized) rewards."""
    fns = [make_env(0, start_level=1, max_level=NUM_LEVELS,
                    enable_powerups=True, seed=42)]
    venv = DummyVecEnv(fns)
    return VecNormalize(venv, norm_obs=False, norm_reward=False)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class CurriculumCallback(BaseCallback):
    """Advances curriculum phases based on CURRICULUM timestep schedule."""

    def __init__(self, curriculum, verbose: int = 0):
        super().__init__(verbose)
        self.curriculum = sorted(curriculum, key=lambda x: x[0])
        self.current_phase = 0

    def _on_step(self) -> bool:
        new_phase = self.current_phase
        for i, (start_ts, *_) in enumerate(self.curriculum):
            if self.num_timesteps >= start_ts:
                new_phase = i

        if new_phase != self.current_phase:
            self.current_phase = new_phase
            _, min_lvl, max_lvl, powerups = self.curriculum[new_phase]
            if self.verbose:
                print(f"[Curriculum] Phase {new_phase}: levels {min_lvl}–{max_lvl}, "
                      f"powerups={'ON' if powerups else 'OFF'} @ step {self.num_timesteps:,}")
            env = self.training_env
            while hasattr(env, "venv"):
                env = env.venv
            if hasattr(env, "env_method"):
                env.env_method("set_curriculum",
                               start_level=min_lvl,
                               max_level=max_lvl,
                               enable_powerups=powerups)
        return True


class EntropyScheduleCallback(BaseCallback):
    """Linearly decays ent_coef from start → end over the learn() call."""

    def __init__(self, start: float, end: float, total_timesteps: int,
                 verbose: int = 0):
        super().__init__(verbose)
        self.ent_start = start
        self.ent_end = end
        self.total = total_timesteps
        self._step0 = None

    def _on_training_start(self) -> None:
        self._step0 = self.num_timesteps

    def _on_step(self) -> bool:
        elapsed = self.num_timesteps - (self._step0 or 0)
        frac = min(1.0, elapsed / max(1, self.total))
        self.model.ent_coef = self.ent_start + (self.ent_end - self.ent_start) * frac
        return True


class MetricsCallback(BaseCallback):
    """Logs per-episode game stats to TensorBoard at episode boundaries."""

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep = info["episode"]
                self.logger.record("game/episode_reward", ep["r"])
                self.logger.record("game/episode_length", ep["l"])
                self.logger.record("game/levels_cleared", info.get("levels_cleared", 0))
                self.logger.record("game/highest_level", info.get("highest_level", 1))
                self.logger.record("game/game_cleared", 1 if info.get("game_cleared") else 0)
                self.logger.record("game/shots_wasted", info.get("shots_wasted", 0))
                self.logger.record("game/ceiling_pops", info.get("ceiling_pops", 0))
        return True


def make_callbacks(checkpoint_dir: str, log_dir: str, eval_env,
                   n_envs: int, phase_name: str,
                   curriculum=None, ent_schedule=None) -> CallbackList:
    cbs = [
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(checkpoint_dir, "best"),
            log_path=os.path.join(log_dir, "eval"),
            eval_freq=max(500_000 // n_envs, 1000),
            n_eval_episodes=10,
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=max(5_000_000 // n_envs, 1000),
            save_path=os.path.join(checkpoint_dir, "periodic"),
            name_prefix=f"ppo_{phase_name}",
            verbose=1,
        ),
        MetricsCallback(),
    ]
    if curriculum is not None:
        cbs.append(CurriculumCallback(curriculum, verbose=1))
    if ent_schedule is not None:
        cbs.append(ent_schedule)
    return CallbackList(cbs)


# ---------------------------------------------------------------------------
# Schedules & model construction
# ---------------------------------------------------------------------------

def linear_schedule(start: float, end: float):
    """Linear interpolation: start at progress_remaining=1, end at 0."""
    def schedule(progress_remaining: float) -> float:
        return end + (start - end) * progress_remaining
    return schedule


def _adjusted_batch_size(n_envs: int) -> int:
    """Ensure batch_size divides evenly into n_envs * n_steps."""
    buffer = n_envs * T["n_steps"]
    batch = min(T["batch_size"], buffer)
    while buffer % batch != 0 and batch > 1:
        batch //= 2
    return batch


def build_model(env, device: str, seed: int | None,
                log_dir: str) -> MaskablePPO:
    n_envs = env.num_envs
    return MaskablePPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=linear_schedule(T["learning_rate_start"], T["learning_rate_end"]),
        n_steps=T["n_steps"],
        batch_size=_adjusted_batch_size(n_envs),
        n_epochs=T["n_epochs"],
        gamma=T["gamma"],
        gae_lambda=T["gae_lambda"],
        clip_range=T["clip_range"],
        ent_coef=T["ent_coef_start"],  # decayed by EntropyScheduleCallback
        vf_coef=T["vf_coef"],
        max_grad_norm=T["max_grad_norm"],
        policy_kwargs=dict(
            net_arch=dict(pi=T["net_arch_pi"], vf=T["net_arch_vf"]),
            activation_fn=th.nn.ReLU,
        ),
        tensorboard_log=log_dir,
        verbose=1,
        device=device,
        seed=seed,
    )


def save(model, env, checkpoint_dir: str, name: str) -> tuple[str, str]:
    model_path = os.path.join(checkpoint_dir, name)
    model.save(model_path)
    vecnorm = env
    while not isinstance(vecnorm, VecNormalize):
        vecnorm = vecnorm.venv
    vecnorm_path = os.path.join(checkpoint_dir, f"{name}_vecnorm.pkl")
    vecnorm.save(vecnorm_path)
    return model_path, vecnorm_path


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train Bubble Trouble with MaskablePPO (two-phase: warmup + level-distributed)"
    )
    parser.add_argument("--timesteps", type=int, default=None,
                        help=f"Total training timesteps (default: {T['total_timesteps']:,})")
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="Phase 1 warmup timesteps (default: 40%% of total)")
    parser.add_argument("--warmup-only", action="store_true",
                        help="Run only Phase 1 (curriculum warmup)")
    parser.add_argument("--skip-warmup", action="store_true",
                        help="Skip to Phase 2 (requires --resume)")
    parser.add_argument("--n-envs", type=int, default=None,
                        help=f"Parallel envs (default: {T['n_envs']} warmup, {NUM_LEVELS} mastery)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .zip to resume from")
    parser.add_argument("--vecnorm", type=str, default=None,
                        help="VecNormalize stats .pkl path when resuming")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Device (default: cpu)")
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = args.device
    if device == "auto":
        device = "cuda" if th.cuda.is_available() else "cpu"

    total_timesteps = args.timesteps or T["total_timesteps"]
    n_warmup_envs = args.n_envs or T["n_envs"]
    n_mastery_envs = args.n_envs or NUM_LEVELS
    warmup_steps = args.warmup_steps or int(total_timesteps * 0.4)
    mastery_steps = total_timesteps - warmup_steps

    vecnorm_path = args.vecnorm  # track across phases

    # ===================================================================
    # Phase 1: Curriculum Warmup
    # ===================================================================
    if not args.skip_warmup:
        print(f"\n{'='*60}")
        print(f"Phase 1: Curriculum Warmup  ({warmup_steps:,} steps, {n_warmup_envs} envs)")
        print(f"  net={T['net_arch_pi']}  gamma={T['gamma']}  device={device}")
        print(f"  LR: {T['learning_rate_start']:.1e} → {T['learning_rate_end']:.1e}")
        print(f"  entropy: {T['ent_coef_start']} → {T['ent_coef_end']}")
        print(f"{'='*60}\n")

        _, min_lvl, max_lvl, powerups = CURRICULUM[0]
        env = build_env(n_warmup_envs, min_lvl, max_lvl, powerups, args.seed)
        eval_env = build_eval_env()

        if args.resume:
            print(f"Resuming warmup from {args.resume}")
            if vecnorm_path and os.path.exists(vecnorm_path):
                env = VecNormalize.load(vecnorm_path, env.venv)
                env.norm_obs = False
                env.clip_reward = 50.0
            model = MaskablePPO.load(args.resume, env=env, device=device)
        else:
            model = build_model(env, device, args.seed, args.log_dir)

        ent_cb = EntropyScheduleCallback(
            T["ent_coef_start"], T["ent_coef_end"], warmup_steps)

        model.learn(
            total_timesteps=warmup_steps,
            callback=make_callbacks(
                args.checkpoint_dir, args.log_dir, eval_env,
                n_warmup_envs, "warmup",
                curriculum=CURRICULUM, ent_schedule=ent_cb,
            ),
            progress_bar=True,
            reset_num_timesteps=(args.resume is None),
            log_interval=100,
        )

        warmup_model_path, warmup_vecnorm_path = save(
            model, env, args.checkpoint_dir, "warmup_model"
        )
        print(f"\nPhase 1 complete → {warmup_model_path}.zip")
        env.close()
        eval_env.close()

        if args.warmup_only:
            print("--warmup-only: stopping after Phase 1.")
            return

        resume_path = warmup_model_path + ".zip"
        vecnorm_path = warmup_vecnorm_path
    else:
        if not args.resume:
            print("ERROR: --skip-warmup requires --resume <checkpoint.zip>")
            sys.exit(1)
        resume_path = args.resume

    # ===================================================================
    # Phase 2: Level-Distributed Mastery
    # ===================================================================
    print(f"\n{'='*60}")
    print(f"Phase 2: Level-Distributed Mastery  ({mastery_steps:,} steps, {n_mastery_envs} envs)")
    print(f"  {n_mastery_envs} envs across {NUM_LEVELS} levels (round-robin)")
    print(f"  LR: {T['learning_rate_start']*0.3:.1e} → {T['learning_rate_end']:.1e}")
    print(f"  entropy: {T['ent_coef_end']}  (fixed, exploitation-focused)")
    print(f"{'='*60}\n")

    env = build_level_distributed_env(n_mastery_envs, NUM_LEVELS, args.seed)
    if vecnorm_path and os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env.venv)
        env.norm_obs = False
        env.clip_reward = 50.0
    eval_env = build_eval_env()

    # Lower LR for fine-tuning; fixed low entropy for exploitation
    lr_mastery = linear_schedule(T["learning_rate_start"] * 0.3,
                                 T["learning_rate_end"])
    batch_size = _adjusted_batch_size(n_mastery_envs)

    model = MaskablePPO.load(
        resume_path,
        env=env,
        device=device,
        custom_objects={
            "learning_rate": lr_mastery,
            "ent_coef": T["ent_coef_end"],
            "batch_size": batch_size,
        },
    )

    model.learn(
        total_timesteps=mastery_steps,
        callback=make_callbacks(
            args.checkpoint_dir, args.log_dir, eval_env,
            n_mastery_envs, "mastery",
        ),
        progress_bar=True,
        reset_num_timesteps=False,
        log_interval=100,
    )

    model_path, final_vecnorm = save(model, env, args.checkpoint_dir, "final_model")
    print(f"\nTraining complete — saved to {model_path}.zip")
    print(f"VecNormalize stats → {final_vecnorm}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
