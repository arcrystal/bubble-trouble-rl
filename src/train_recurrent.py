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
  - MlpLstmPolicy: 512×256 MLP + 256-unit LSTM + separate critic LSTM
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
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.monitor import Monitor

from bubble_env import BubbleTroubleEnv
from config import RECURRENT_TRAINING as T, RECURRENT_CURRICULUM as CURRICULUM, NUM_LEVELS


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
    print(f"  curriculum: {len(CURRICULUM)} phases")
    print(f"  device={device}")
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
        model = RecurrentPPO.load(args.resume, env=env, device=device)
    else:
        model = build_model(env, device, args.seed, args.log_dir)

    # --- Callbacks ---
    ent_cb = EntropyScheduleCallback(
        T["ent_coef_start"], T["ent_coef_end"], total_timesteps)

    callbacks = CallbackList([
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(args.checkpoint_dir, "best"),
            log_path=os.path.join(args.log_dir, "eval_recurrent"),
            eval_freq=max(100_000 // n_envs, 1000),
            n_eval_episodes=10,
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=max(5_000_000 // n_envs, 1000),
            save_path=os.path.join(args.checkpoint_dir, "periodic"),
            name_prefix="recurrent_ppo",
            verbose=1,
        ),
        MetricsCallback(),
        CurriculumCallback(CURRICULUM, verbose=1),
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
