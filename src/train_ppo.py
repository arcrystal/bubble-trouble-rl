"""Bubble Trouble RL — MaskablePPO with cycling curriculum.

Single continuous training run with a cycling curriculum:
  1. Progressive expansion (35%): levels 1-3 → 1-5 → 3-8 → 5-10 → 7-12
  2. Full game + hard refreshers (65%): full 1-12, then hard 5-12, then final 1-12

All envs always play the SAME level range — coherent batches, no distribution
shock. The full-game phases match the evaluation protocol exactly.
Hard refresher phases force practice on levels the agent rarely reaches
organically, without corrupting the value function.

Key mechanisms:
  - target_kl: Stops gradient updates early if the policy changes too fast.
  - gamma=0.999: ~1000-step planning horizon for ceiling pop chains.
  - 6 gradient epochs per rollout with target_kl as guardrail.
  - DeepSets feature extractor + 512×256 MLP policy/value heads.

Usage:
  python src/train_ppo.py                                      # full 500M steps
  python src/train_ppo.py --timesteps 1000000 --n-envs 4       # quick smoke test
  python src/train_ppo.py --resume checkpoints/ppo/periodic/ppo_XXX.zip --vecnorm ...
  python src/train_ppo.py --warmup checkpoints/user_warmups/bc_pretrained.zip \\
      --demo checkpoints/user_warmups/demo_*.npz               # BC warm start + DAPG
"""

import argparse
import json
import os
import resource
from datetime import datetime

import torch as th

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor

from bubble_env import BubbleTroubleEnv
from config import TRAINING as T, CURRICULUM, WARMUP_CURRICULUM, NUM_LEVELS
from feature_extractor import BubbleFeatureExtractor
from callbacks import (
    BCAuxiliaryCallback, BestMeanEvalCallback, BestVecNormalizeCallback,
    CurriculumCallback, DeathCreditCallback, EntropyScheduleCallback,
    MetricsCallback, PolicyAnchorCallback, SelfImitationCallback,
    VecNormalizeCheckpointCallback,
)


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
    """All workers see the same level range."""
    fns = [make_env(i, start_level, max_level, powerups, seed)
           for i in range(n_envs)]
    venv = SubprocVecEnv(fns, start_method="fork") if n_envs > 1 else DummyVecEnv(fns)
    return VecNormalize(venv, norm_obs=False, norm_reward=True, clip_reward=150.0)


N_EVAL_ENVS = 8

def build_eval_env() -> VecNormalize:
    """Eval env: 8 parallel full-game workers, each with a distinct seed."""
    fns = [make_env(i, start_level=1, max_level=NUM_LEVELS,
                    enable_powerups=True, seed=42 + i) for i in range(N_EVAL_ENVS)]
    venv = SubprocVecEnv(fns, start_method="fork")
    return VecNormalize(venv, norm_obs=False, norm_reward=False)


# ---------------------------------------------------------------------------
# Schedules & model construction
# ---------------------------------------------------------------------------

def linear_schedule(start: float, end: float):
    """Linear interpolation: start at progress_remaining=1, end at 0."""
    def schedule(progress_remaining: float) -> float:
        return end + (start - end) * progress_remaining
    return schedule


def log_linear_schedule(start: float, end: float):
    """Exponential (log-linear) decay: log(LR) is linear in time.
    Drops LR quickly early on, then fine-tunes slowly for most of training.
    """
    import math
    log_start = math.log(start)
    log_end = math.log(end)
    def schedule(progress_remaining: float) -> float:
        return math.exp(log_end + (log_start - log_end) * progress_remaining)
    return schedule



def _adjusted_batch_size(n_envs: int) -> int:
    """Ensure batch_size divides evenly into n_envs * n_steps."""
    buffer = n_envs * T["n_steps"]
    batch = min(T["batch_size"], buffer)
    while buffer % batch != 0 and batch > 1:
        batch //= 2
    return batch


def _reset_value_head(policy):
    """Reinitialize value function layers, keeping actor + features extractor.

    BC's value head was trained on z-scored MC returns: (G - mean) / std.
    VecNormalize normalizes rewards differently: r / sqrt(return_var), no
    mean subtraction. The scale mismatch causes massive value-loss gradients
    that corrupt the shared features extractor — worse than random init.

    Uses SB3's default orthogonal init: hidden gain=sqrt(2), output gain=1.
    """
    import numpy as np
    for layer in policy.mlp_extractor.value_net.modules():
        if isinstance(layer, th.nn.Linear):
            th.nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            th.nn.init.zeros_(layer.bias)
    for layer in policy.value_net.modules():
        if isinstance(layer, th.nn.Linear):
            th.nn.init.orthogonal_(layer.weight, gain=1.0)
            th.nn.init.zeros_(layer.bias)


def _seed_vecnormalize(model, env, n_steps=2048):
    """Collect a short rollout with BC policy to calibrate VecNormalize.

    Without seeding, ret_rms.var=1 means the first real PPO rollout has
    un-scaled rewards, amplifying the advantage estimation noise.
    """
    obs = env.reset()
    for _ in range(n_steps):
        with th.no_grad():
            actions, _ = model.predict(obs, deterministic=False)
        obs, _, _, _ = env.step(actions)
    env.reset()


def build_model(env, device: str, seed: int | None,
                log_dir: str) -> MaskablePPO:
    n_envs = env.num_envs
    return MaskablePPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=log_linear_schedule(T["learning_rate_start"], T["learning_rate_end"]),
        n_steps=T["n_steps"],
        batch_size=_adjusted_batch_size(n_envs),
        n_epochs=T["n_epochs"],
        gamma=T["gamma"],
        gae_lambda=T["gae_lambda"],
        clip_range=T["clip_range"],
        ent_coef=T["ent_coef_start"],  # decayed by EntropyScheduleCallback
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_eval_thresholds(eval_cb, best_dir: str) -> None:
    """Restore best_mean_reward threshold so a resumed run doesn't clobber existing best models.

    On a fresh run the eval callback starts with best_mean_reward = -inf, so
    the very first evaluation always triggers a save regardless of quality.
    Loading the persisted score prevents that false positive on resume.
    """
    score_path = os.path.join(best_dir, "best_score.json")
    if not os.path.exists(score_path):
        return
    with open(score_path) as f:
        score = json.load(f)
    eval_cb.best_mean_reward = score["best_mean_reward"]
    print(f"[Resume] Seeded eval threshold from {score_path}: best_mean={score['best_mean_reward']:.2f}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    # Raise the open-file-descriptor limit to handle many SubprocVecEnv workers.
    # macOS terminal sessions inherit a low soft limit from the shell.
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 65536), hard))

    parser = argparse.ArgumentParser(
        description="Train Bubble Trouble with MaskablePPO (cycling curriculum)"
    )
    parser.add_argument("--timesteps", type=int, default=None,
                        help=f"Total training timesteps (default: {T['total_timesteps']:,})")
    parser.add_argument("--n-envs", type=int, default=None,
                        help=f"Parallel envs (default: {T['n_envs']})")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .zip to resume from")
    parser.add_argument("--vecnorm", type=str, default=None,
                        help="VecNormalize stats .pkl path when resuming")
    parser.add_argument("--warmup", type=str, default=None,
                        help="Path to BC-pretrained model .zip for warm start")
    parser.add_argument("--demo", type=str, nargs="+", default=None,
                        help="Demo .npz files for DAPG auxiliary loss during training "
                             "(use with --warmup for best results)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate for this run (constant float, e.g. 3e-5)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Device (default: cpu)")
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/ppo",
                        help="Base checkpoints dir; a timestamped run subdir is created automatically")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Explicit run directory (overrides auto-generated name, useful for --resume)")
    args = parser.parse_args()

    # Each run gets its own directory; --resume should pass --run-dir to reuse it
    if args.run_dir:
        run_dir = args.run_dir
    else:
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir = os.path.join(args.checkpoint_dir, run_name)

    periodic_dir = os.path.join(run_dir, "periodic")
    best_dir = os.path.join(run_dir, "best")
    os.makedirs(periodic_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print(f"Run directory: {run_dir}")

    device = args.device
    if device == "auto":
        device = "cuda" if th.cuda.is_available() else "cpu"

    if args.resume and args.warmup:
        parser.error("--resume and --warmup are mutually exclusive")

    total_timesteps = args.timesteps or T["total_timesteps"]
    n_envs = args.n_envs or T["n_envs"]

    # Select curriculum based on warmup mode
    curriculum = WARMUP_CURRICULUM if args.warmup else CURRICULUM

    lr_start = T["learning_rate_start"]
    ent_start = T["ent_coef_start"]
    target_kl = T["target_kl"]
    if args.warmup:
        lr_start = T["learning_rate_start"] / 10  # 10x lower for warm-started policy
        ent_start = 0.005  # lower entropy — agent already has a reasonable policy
        target_kl = 0.05  # BC policy is peaked → small param changes cause large KL;
                          # clip_range is the real update guard, target_kl is secondary

    print(f"\n{'='*60}")
    print(f"MaskablePPO — {'Warmup' if args.warmup else 'Cycling'} Curriculum  "
          f"({total_timesteps:,} steps, {n_envs} envs)")
    print(f"  net={T['net_arch_pi']}  gamma={T['gamma']}  target_kl={target_kl}")
    print(f"  n_steps={T['n_steps']}  batch={T['batch_size']}  epochs={T['n_epochs']}")
    print(f"  LR: {lr_start:.1e} → {T['learning_rate_end']:.1e} (log-linear)")
    print(f"  entropy: {ent_start} → {T['ent_coef_end']}")
    print(f"  device={device}")
    print(f"\n  {'Phase':<6} {'Step':>12} {'Levels':>8} {'Powerups':>9}")
    print(f"  {'-'*6} {'-'*12} {'-'*8} {'-'*9}")
    for i, (start_ts, min_l, max_l, pups) in enumerate(curriculum):
        print(f"  {i:<6} {start_ts:>12,} {f'{min_l}-{max_l}':>8} {'ON' if pups else 'OFF':>9}")
    print(f"{'='*60}\n")

    # --- Build env from first curriculum phase ---
    _, min_lvl, max_lvl, powerups = curriculum[0]
    env = build_env(n_envs, min_lvl, max_lvl, powerups, args.seed)
    eval_env = build_eval_env()

    lr = args.lr if args.lr is not None else log_linear_schedule(lr_start, T["learning_rate_end"])

    if args.resume:
        print(f"Resuming from {args.resume}")
        if args.lr is not None:
            print(f"  LR override: {args.lr:.2e} (constant)")
        if args.vecnorm and os.path.exists(args.vecnorm):
            env = VecNormalize.load(args.vecnorm, env.venv)
            env.norm_obs = False
            env.clip_reward = 150.0
        else:
            print("[WARNING] Resuming without VecNormalize stats — reward normalization will reset")
        model = MaskablePPO.load(args.resume, env=env, device=device, learning_rate=lr)
        # Restore curriculum to the correct phase based on saved timestep
        resume_phase = 0
        for i, (start_ts, *_) in enumerate(curriculum):
            if model.num_timesteps >= start_ts:
                resume_phase = i
        _, min_lvl, max_lvl, powerups = curriculum[resume_phase]
        inner_env = env
        while hasattr(inner_env, "venv"):
            inner_env = inner_env.venv
        inner_env.env_method("set_curriculum",
                             start_level=min_lvl, max_level=max_lvl,
                             enable_powerups=powerups)
        print(f"[Resume] Phase {resume_phase}: levels {min_lvl}–{max_lvl}, "
              f"powerups={'ON' if powerups else 'OFF'} @ step {model.num_timesteps:,}")
    elif args.warmup:
        model = build_model(env, device, args.seed, args.log_dir)
        warmup = MaskablePPO.load(args.warmup, device=device)
        model.policy.load_state_dict(warmup.policy.state_dict())
        del warmup
        # Override LR schedule for warm start (must set lr_schedule too —
        # SB3 caches the schedule during _setup_model and uses lr_schedule, not learning_rate)
        warmup_schedule = lr if args.lr is not None else log_linear_schedule(lr_start, T["learning_rate_end"])
        model.learning_rate = warmup_schedule
        model.lr_schedule = warmup_schedule
        model.ent_coef = ent_start
        model.target_kl = target_kl
        # Reset value head: BC value was trained on z-scored MC returns which
        # don't match VecNormalize's reward scaling. Random-init V≈0 gives
        # unbiased advantages; miscalibrated V gives biased ones that corrupt
        # the shared features extractor via large value-loss gradients.
        _reset_value_head(model.policy)
        # Seed VecNormalize so ret_rms.var is calibrated before the first
        # real PPO rollout (prevents wild advantage estimates at step 0).
        print(f"[Warmup] Loaded BC actor from {args.warmup}, value head reset")
        print(f"[Warmup] LR={lr_start:.1e}, ent_coef={ent_start}, target_kl={target_kl}")
        print("[Warmup] Seeding VecNormalize reward statistics...")
        _seed_vecnormalize(model, env)
        print("[Warmup] VecNormalize seeded")
    else:
        model = build_model(env, device, args.seed, args.log_dir)

    # --- Callbacks ---
    ent_cb = EntropyScheduleCallback(ent_start, T["ent_coef_end"], total_timesteps)

    eval_cb = BestMeanEvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=os.path.join(args.log_dir, "eval_ppo"),
        eval_freq=T["n_steps"],  # every rollout
        n_eval_episodes=N_EVAL_ENVS,
        deterministic=True,
        callback_on_new_best=BestVecNormalizeCallback(
            save_path=best_dir,
            verbose=1,
        ),
        verbose=1,
    )
    if args.resume:
        _seed_eval_thresholds(eval_cb, best_dir)
    cb_list = [
        eval_cb,
        VecNormalizeCheckpointCallback(
            save_freq=max(50_000_000 // n_envs, 1000),
            save_path=periodic_dir,
            name_prefix="ppo",
            verbose=1,
        ),
        MetricsCallback(),
        CurriculumCallback(curriculum, eval_callback=eval_cb,
                           entropy_callback=ent_cb, verbose=1),
        ent_cb,
        DeathCreditCallback(),
        SelfImitationCallback(verbose=1),
    ]

    # KL anchoring: prevent policy drift from BC on the RL state distribution.
    # Unlike DAPG (which uses demo observations), this operates on rollout data
    # — no distribution mismatch. Requires --warmup (snapshots the BC policy).
    if args.warmup:
        anchor_cb = PolicyAnchorCallback(
            kl_coef_start=0.5,
            total_timesteps=total_timesteps,
            n_steps=8,
            batch_size=512,
            lr_ratio=0.5,
            verbose=1,
        )
        cb_list.append(anchor_cb)

    # DAPG: auxiliary BC loss to prevent catastrophic forgetting of demo skills
    if args.demo:
        bc_cb = BCAuxiliaryCallback(
            demo_paths=args.demo,
            curriculum=curriculum,
            bc_weight_start=1.0,
            total_timesteps=total_timesteps,
            bc_lr_ratio=1.0,
            n_dapg_steps=16,
            batch_size=256,
            verbose=1,
        )
        cb_list.append(bc_cb)
        print(f"[DAPG] BC auxiliary loss enabled with {len(args.demo)} demo pattern(s)")

    callbacks = CallbackList(cb_list)

    # --- Train ---
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=(args.resume is None),
        log_interval=100,
    )

    model_path, vecnorm_path = save(model, env, run_dir, "final_model")
    print(f"\nTraining complete — saved to {model_path}.zip")
    print(f"VecNormalize stats → {vecnorm_path}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
