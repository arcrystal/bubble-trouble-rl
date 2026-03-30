"""Bubble Trouble RL — MaskablePPO for Infinity Mode (survival).

Trains a MaskablePPO agent on InfinityModeEnv with action masking. No
curriculum — difficulty ramps naturally via the engine's time-based spawning.

Reward: full base engine reward function (same as regular mode for model interchangeability).

Usage:
  python src/train_infinity.py                                    # full run
  python src/train_infinity.py --timesteps 1000000 --n-envs 4     # quick smoke test
  python src/train_infinity.py --resume checkpoints/infinity/periodic/infinity_XXX.zip \
      --vecnorm checkpoints/infinity/periodic/infinity_XXX_vecnorm.pkl
"""

import argparse
import json
import os
import resource

import torch as th

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor

from infinity_env import InfinityModeEnv
from config import INFINITY_TRAINING as T
from feature_extractor import BubbleFeatureExtractor
from stable_baselines3.common.callbacks import EvalCallback
from callbacks import (
    BestVecNormalizeCallback, EntropyScheduleCallback,
    MetricsCallback, VecNormalizeCheckpointCallback,
)


# ---------------------------------------------------------------------------
# Environment factories
# ---------------------------------------------------------------------------

def _mask_fn(env):
    return env.action_masks()


def make_env(rank: int, seed: int | None = None):
    def _init():
        env = InfinityModeEnv(seed=None)  # random seed each episode for training
        env = ActionMasker(env, _mask_fn)
        env = Monitor(env)
        env.reset(seed=(seed + rank) if seed is not None else None)
        return env
    return _init


def build_env(n_envs: int, seed: int | None) -> VecNormalize:
    fns = [make_env(i, seed) for i in range(n_envs)]
    venv = SubprocVecEnv(fns, start_method="fork") if n_envs > 1 else DummyVecEnv(fns)
    return VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=150.0)


N_EVAL_ENVS = 8

def build_eval_env() -> VecNormalize:
    """Eval env: 8 parallel workers with fixed seeds for reproducible evaluation."""
    def _make_eval(rank):
        def _init():
            env = InfinityModeEnv(seed=42 + rank)
            env = ActionMasker(env, _mask_fn)
            env = Monitor(env)
            env.reset()
            return env
        return _init

    fns = [_make_eval(i) for i in range(N_EVAL_ENVS)]
    venv = SubprocVecEnv(fns, start_method="fork")
    return VecNormalize(venv, norm_obs=False, norm_reward=False)


# ---------------------------------------------------------------------------
# Schedules & model construction
# ---------------------------------------------------------------------------

def linear_schedule(start: float, end: float):
    def schedule(progress_remaining: float) -> float:
        return end + (start - end) * progress_remaining
    return schedule


def _adjusted_batch_size(n_envs: int) -> int:
    buffer = n_envs * T["n_steps"]
    batch = min(T["batch_size"], buffer)
    while buffer % batch != 0 and batch > 1:
        batch //= 2
    return batch


def _reset_value_head(policy):
    """Reinitialize value function layers, keeping actor + features extractor."""
    import numpy as np
    for layer in policy.mlp_extractor.value_net.modules():
        if isinstance(layer, th.nn.Linear):
            th.nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            th.nn.init.zeros_(layer.bias)
    for layer in policy.value_net.modules():
        if isinstance(layer, th.nn.Linear):
            th.nn.init.orthogonal_(layer.weight, gain=1.0)
            th.nn.init.zeros_(layer.bias)


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


def _seed_eval_thresholds(eval_cb, best_dir: str) -> None:
    """Restore best-score thresholds so a resumed run doesn't clobber best_model."""
    score_path = os.path.join(best_dir, "best_score.json")
    if not os.path.exists(score_path):
        return
    with open(score_path) as f:
        score = json.load(f)
    eval_cb.best_mean_reward = score["best_mean_reward"]
    if hasattr(eval_cb, "best_max_reward") and "best_max_reward" in score:
        eval_cb.best_max_reward = score["best_max_reward"]
    print(f"[Resume] Seeded eval thresholds from {score_path}: "
          f"best_mean={score['best_mean_reward']:.2f}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 65536), hard))

    parser = argparse.ArgumentParser(
        description="Train Bubble Trouble Infinity Mode with MaskablePPO (survival)"
    )
    parser.add_argument("--timesteps", type=int, default=None,
                        help=f"Total training timesteps (default: {T['total_timesteps']:,})")
    parser.add_argument("--n-envs", type=int, default=None,
                        help=f"Parallel envs (default: {T['n_envs']})")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to MaskablePPO checkpoint .zip to resume from")
    parser.add_argument("--warmup", type=str, default=None,
                        help="Path to any model .zip to warm-start from "
                             "(transfers weights, resets optimizer/timesteps)")
    parser.add_argument("--vecnorm", type=str, default=None,
                        help="VecNormalize stats .pkl path when resuming")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate for this run (constant float, e.g. 3e-5)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--log-dir", type=str, default="./logs/infinity")
    parser.add_argument("--checkpoint-dir", type=str,
                        default="./checkpoints/infinity")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = args.device
    if device == "auto":
        device = "cuda" if th.cuda.is_available() else "cpu"

    if args.resume and args.warmup:
        parser.error("--resume and --warmup are mutually exclusive")

    total_timesteps = args.timesteps or T["total_timesteps"]
    n_envs = args.n_envs or T["n_envs"]

    print(f"\n{'='*60}")
    print(f"MaskablePPO — Infinity Mode (survival)  "
          f"({total_timesteps:,} steps, {n_envs} envs)")
    print(f"  net={T['net_arch_pi']}  gamma={T['gamma']}  target_kl={T['target_kl']}")
    print(f"  n_steps={T['n_steps']}  batch={T['batch_size']}  epochs={T['n_epochs']}")
    print(f"  LR: {T['learning_rate_start']:.1e} → {T['learning_rate_end']:.1e}")
    print(f"  entropy: {T['ent_coef_start']} → {T['ent_coef_end']}")
    print(f"  device={device}")
    print(f"{'='*60}\n")

    env = build_env(n_envs, args.seed)
    eval_env = build_eval_env()

    lr = args.lr if args.lr is not None else linear_schedule(T["learning_rate_start"], T["learning_rate_end"])

    if args.resume:
        print(f"Resuming from {args.resume}")
        if args.lr is not None:
            print(f"  LR override: {args.lr:.2e} (constant)")
        if args.vecnorm and os.path.exists(args.vecnorm):
            env = VecNormalize.load(args.vecnorm, env.venv)
            env.norm_obs = False
            env.clip_reward = 150.0
        else:
            print("[WARNING] Resuming without VecNormalize stats — "
                  "reward normalization will reset")
        model = MaskablePPO.load(args.resume, env=env, device=device, learning_rate=lr)
    elif args.warmup:
        # Both regular and infinity modes share the same obs space (968-dim),
        # so weights transfer directly with no shape mismatches.
        model = build_model(env, device, args.seed, args.log_dir)
        source = MaskablePPO.load(args.warmup, device=device)
        model.policy.load_state_dict(source.policy.state_dict())
        del source
        # Reset value head — source's value function was calibrated for regular
        # mode rewards which may differ in scale from infinity mode.
        _reset_value_head(model.policy)
        # Gentle LR for warm start (10x lower), unless --lr explicitly overrides
        warmup_lr_start = T["learning_rate_start"] / 10
        warmup_schedule = (lr if args.lr is not None
                           else linear_schedule(warmup_lr_start, T["learning_rate_end"]))
        model.learning_rate = warmup_schedule
        model.lr_schedule = warmup_schedule
        model.ent_coef = T["ent_coef_start"] * 0.5  # lower entropy for warm start
        lr_end = T["learning_rate_end"]
        lr_desc = "constant" if args.lr else f"{warmup_lr_start:.1e}→{lr_end:.1e}"
        print(f"[Warmup] Loaded weights from {args.warmup} (value head reset, LR={lr_desc})")
    else:
        model = build_model(env, device, args.seed, args.log_dir)

    # --- Callbacks ---
    ent_cb = EntropyScheduleCallback(
        T["ent_coef_start"], T["ent_coef_end"], total_timesteps)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.checkpoint_dir, "best"),
        log_path=os.path.join(args.log_dir, "eval_infinity"),
        eval_freq=T["n_steps"],
        n_eval_episodes=N_EVAL_ENVS,
        deterministic=True,
        callback_on_new_best=BestVecNormalizeCallback(
            save_path=os.path.join(args.checkpoint_dir, "best"),
            verbose=1,
        ),
        verbose=1,
    )
    if args.resume:
        _seed_eval_thresholds(eval_cb, os.path.join(args.checkpoint_dir, "best"))
    callbacks = CallbackList([
        eval_cb,
        VecNormalizeCheckpointCallback(
            save_freq=max(50_000_000 // n_envs, 1000),
            save_path=os.path.join(args.checkpoint_dir, "periodic"),
            name_prefix="infinity",
            verbose=1,
        ),
        MetricsCallback(),
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

    model_path, vecnorm_path = save(model, env, args.checkpoint_dir,
                                    "final_model")
    print(f"\nTraining complete — saved to {model_path}.zip")
    print(f"VecNormalize stats → {vecnorm_path}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
