"""Bubble Trouble RL — RecurrentPPO (LSTM) with cycling curriculum.

Same curriculum structure as train_ppo.py but uses an LSTM policy for
temporal reasoning. The LSTM learns multi-step strategies natively:
  - Position → wait for ball to rise → shoot for ceiling pop chain
  - Track laser cooldown state across frames
  - Learn level-specific dodge/attack patterns over time

No action masking is available for RecurrentPPO. The LSTM learns to
avoid invalid shots via the can_fire observation feature and the
wasted_shot penalty.

Key differences from MaskablePPO:
  - MlpLstmPolicy: DeepSets extractor + 512×256 MLP + 256-unit LSTM + separate critic LSTM
  - Sequence-aware rollout buffer (preserves hidden states)
  - n_steps=2048: shorter rollouts since recurrent buffer is heavier per step

Usage:
  python src/train_recurrent.py                                  # full 300M steps
  python src/train_recurrent.py --timesteps 1000000 --n-envs 4   # quick smoke test
  python src/train_recurrent.py --resume checkpoints/recurrent/final_model.zip --vecnorm ...
"""

import argparse
import os

import torch as th

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from bubble_env import BubbleTroubleEnv
from config import RECURRENT_TRAINING as T, RECURRENT_CURRICULUM as CURRICULUM, NUM_LEVELS
from feature_extractor import BubbleFeatureExtractor
from callbacks import (
    BestVecNormalizeCallback, CurriculumCallback, EntropyScheduleCallback,
    MetricsCallback, VecNormalizeCheckpointCallback,
)


# ---------------------------------------------------------------------------
# Environment factories
# ---------------------------------------------------------------------------

def make_env(rank: int, start_level: int = 1, max_level: int = NUM_LEVELS,
             enable_powerups: bool = False, seed: int | None = None):
    def _init():
        env = BubbleTroubleEnv(
            start_level=start_level,
            max_level=max_level,
            enable_powerups=enable_powerups,
            sequential_levels=True,
        )
        env = Monitor(env)
        env.reset(seed=(seed + rank) if seed is not None else None)
        return env
    return _init


def build_env(n_envs: int, start_level: int, max_level: int,
              powerups: bool, seed: int | None) -> VecNormalize:
    """All workers see the same level range."""
    fns = [make_env(i, start_level, max_level, powerups, seed) for i in range(n_envs)]
    venv = SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns)
    return VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=50.0)


N_EVAL_ENVS = 8

def build_eval_env() -> VecNormalize:
    """Eval env: 8 parallel full-game workers, each with a distinct seed."""
    fns = [make_env(i, start_level=1, max_level=NUM_LEVELS,
                    enable_powerups=True, seed=42 + i) for i in range(N_EVAL_ENVS)]
    venv = SubprocVecEnv(fns)
    return VecNormalize(venv, norm_obs=False, norm_reward=False)


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
                log_dir: str) -> RecurrentPPO:
    n_envs = env.num_envs
    return RecurrentPPO(
        policy="MlpLstmPolicy",
        env=env,
        learning_rate=linear_schedule(T["learning_rate_start"], T["learning_rate_end"]),
        n_steps=T["n_steps"],
        batch_size=_adjusted_batch_size(n_envs),
        n_epochs=T["n_epochs"],
        gamma=T["gamma"],
        gae_lambda=T["gae_lambda"],
        clip_range=T["clip_range"],
        ent_coef=T["ent_coef_start"],
        vf_coef=T["vf_coef"],
        max_grad_norm=T["max_grad_norm"],
        target_kl=T["target_kl"],
        policy_kwargs=dict(
            features_extractor_class=BubbleFeatureExtractor,
            features_extractor_kwargs=dict(
                per_ball_hidden=T["per_ball_hidden"],
                per_obstacle_hidden=T["per_obstacle_hidden"],
                context_hidden=T["context_hidden"],
                context_output=T["context_output"],
            ),
            net_arch=dict(pi=T["net_arch_pi"], vf=T["net_arch_vf"]),
            activation_fn=th.nn.ReLU,
            lstm_hidden_size=T["lstm_hidden_size"],
            n_lstm_layers=T["n_lstm_layers"],
            shared_lstm=False,
            enable_critic_lstm=True,
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
        description="Train Bubble Trouble with RecurrentPPO/LSTM (cycling curriculum)"
    )
    parser.add_argument("--timesteps", type=int, default=None,
                        help=f"Total training timesteps (default: {T['total_timesteps']:,})")
    parser.add_argument("--n-envs", type=int, default=None,
                        help=f"Parallel envs (default: {T['n_envs']})")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .zip to resume from")
    parser.add_argument("--vecnorm", type=str, default=None,
                        help="VecNormalize stats .pkl path when resuming")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Device (default: cpu)")
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/recurrent")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = args.device
    if device == "auto":
        device = "cuda" if th.cuda.is_available() else "cpu"

    total_timesteps = args.timesteps or T["total_timesteps"]
    n_envs = args.n_envs or T["n_envs"]

    print(f"\n{'='*60}")
    print(f"RecurrentPPO (LSTM) — Cycling Curriculum  ({total_timesteps:,} steps, {n_envs} envs)")
    print(f"  net={T['net_arch_pi']}  lstm={T['lstm_hidden_size']}×{T['n_lstm_layers']}")
    print(f"  gamma={T['gamma']}  target_kl={T['target_kl']}")
    print(f"  n_steps={T['n_steps']}  batch={T['batch_size']}  epochs={T['n_epochs']}")
    print(f"  LR: {T['learning_rate_start']:.1e} → {T['learning_rate_end']:.1e}")
    print(f"  entropy: {T['ent_coef_start']} → {T['ent_coef_end']}")
    print(f"  device={device}")
    print(f"\n  {'Phase':<6} {'Step':>12} {'Levels':>8} {'Powerups':>9}")
    print(f"  {'-'*6} {'-'*12} {'-'*8} {'-'*9}")
    for i, (start_ts, min_l, max_l, pups) in enumerate(CURRICULUM):
        print(f"  {i:<6} {start_ts:>12,} {f'{min_l}-{max_l}':>8} {'ON' if pups else 'OFF':>9}")
    print(f"{'='*60}\n")

    # --- Build env from first curriculum phase ---
    _, min_lvl, max_lvl, powerups = CURRICULUM[0]
    env = build_env(n_envs, min_lvl, max_lvl, powerups, args.seed)
    eval_env = build_eval_env()

    if args.resume:
        print(f"Resuming from {args.resume}")
        if args.vecnorm and os.path.exists(args.vecnorm):
            env = VecNormalize.load(args.vecnorm, env.venv)
            env.norm_obs = False
            env.clip_reward = 50.0
        else:
            print("[WARNING] Resuming without VecNormalize stats — reward normalization will reset")
        model = RecurrentPPO.load(args.resume, env=env, device=device)
        # Restore curriculum to the correct phase based on saved timestep
        resume_phase = 0
        for i, (start_ts, *_) in enumerate(CURRICULUM):
            if model.num_timesteps >= start_ts:
                resume_phase = i
        _, min_lvl, max_lvl, powerups = CURRICULUM[resume_phase]
        inner_env = env
        while hasattr(inner_env, "venv"):
            inner_env = inner_env.venv
        inner_env.env_method("set_curriculum",
                             start_level=min_lvl, max_level=max_lvl,
                             enable_powerups=powerups)
        print(f"[Resume] Phase {resume_phase}: levels {min_lvl}–{max_lvl}, "
              f"powerups={'ON' if powerups else 'OFF'} @ step {model.num_timesteps:,}")
    else:
        model = build_model(env, device, args.seed, args.log_dir)

    # --- Callbacks ---
    ent_cb = EntropyScheduleCallback(
        T["ent_coef_start"], T["ent_coef_end"], total_timesteps)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.checkpoint_dir, "best"),
        log_path=os.path.join(args.log_dir, "eval_recurrent"),
        eval_freq=max(100_000 // n_envs, 1000),
        n_eval_episodes=N_EVAL_ENVS,
        deterministic=True,
        verbose=1,
    )
    callbacks = CallbackList([
        eval_cb,
        BestVecNormalizeCallback(
            eval_callback=eval_cb,
            save_path=os.path.join(args.checkpoint_dir, "best"),
            verbose=1,
        ),
        VecNormalizeCheckpointCallback(
            save_freq=max(25_000_000 // n_envs, 1000),
            save_path=os.path.join(args.checkpoint_dir, "periodic"),
            name_prefix="recurrent_ppo",
            verbose=1,
        ),
        MetricsCallback(),
        CurriculumCallback(CURRICULUM, eval_callback=eval_cb, verbose=1),
        ent_cb,
    ])

    # --- Train ---
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=(args.resume is None),
        log_interval=100,
    )

    model_path, vecnorm_path = save(model, env, args.checkpoint_dir, "final_model")
    print(f"\nTraining complete — saved to {model_path}.zip")
    print(f"VecNormalize stats → {vecnorm_path}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
