"""Shared training callbacks for Bubble Trouble RL."""

import copy
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize, sync_envs_normalization
from stable_baselines3.common.utils import obs_as_tensor

from config import (ENTROPY_HARD_RESET, DEATH_CREDIT_K, DEATH_CREDIT_PER_STEP,
                    SIL_BUFFER_SIZE, SIL_MIN_LEVEL, SIL_BC_WEIGHT, SIL_LR,
                    SIL_ACTIVATE_AT)


class CurriculumCallback(BaseCallback):
    """Advances curriculum phases based on timestep schedule.

    When entering a hard-level phase (min_level >= 9), triggers an entropy
    reset on the entropy callback to give the agent exploration headroom.
    """

    def __init__(self, curriculum, eval_callback=None, entropy_callback=None,
                 verbose: int = 0):
        super().__init__(verbose)
        self.curriculum = sorted(curriculum, key=lambda x: x[0])
        self.current_phase = 0
        self.eval_callback = eval_callback
        self.entropy_callback = entropy_callback

    def _on_step(self) -> bool:
        new_phase = self.current_phase
        for i, (start_ts, *_) in enumerate(self.curriculum):
            if self.num_timesteps >= start_ts:
                new_phase = i

        if new_phase != self.current_phase:
            old_phase = self.current_phase
            self.current_phase = new_phase
            _, min_lvl, max_lvl, powerups = self.curriculum[new_phase]
            if self.verbose:
                print(f"[Curriculum] Phase {new_phase}: levels {min_lvl}–{max_lvl}, "
                      f"powerups={'ON' if powerups else 'OFF'} "
                      f"@ step {self.num_timesteps:,}")
            env = self.training_env
            while hasattr(env, "venv"):
                env = env.venv
            if hasattr(env, "env_method"):
                env.env_method("set_curriculum",
                               start_level=min_lvl,
                               max_level=max_lvl,
                               enable_powerups=powerups)

            # Entropy reset on hard-level phases
            old_min = self.curriculum[old_phase][1]
            if min_lvl >= 9 and old_min < 9 and self.entropy_callback is not None:
                # Find next phase start for decay target
                next_ts = self.curriculum[new_phase + 1][0] if new_phase + 1 < len(self.curriculum) else None
                self.entropy_callback.set_override(
                    ENTROPY_HARD_RESET, self.num_timesteps, next_ts)
                if self.verbose:
                    print(f"[Curriculum] Entropy reset to {ENTROPY_HARD_RESET} for hard phase")

        _, min_l, max_l, _ = self.curriculum[self.current_phase]
        self.logger.record("train/curriculum_phase", f"{self.current_phase}: {min_l}-{max_l}")
        if self.eval_callback and hasattr(self.eval_callback, "best_mean_reward"):
            self.logger.record("eval/best_mean_reward", self.eval_callback.best_mean_reward)
        return True


class EntropyScheduleCallback(BaseCallback):
    """Linearly decays ent_coef from start to end over total_timesteps.

    Supports mid-training entropy overrides for hard-level curriculum phases.
    When an override is active, entropy decays from the override value back to
    the global schedule value over the override duration.
    """

    def __init__(self, start: float, end: float, total_timesteps: int,
                 verbose: int = 0):
        super().__init__(verbose)
        self.ent_start = start
        self.ent_end = end
        self.total = total_timesteps
        # Override state
        self._override_start_ts = None
        self._override_end_ts = None
        self._override_ent = None

    def set_override(self, ent_value: float, start_ts: int, end_ts: int | None):
        """Set a temporary entropy override that decays to the global schedule."""
        self._override_ent = ent_value
        self._override_start_ts = start_ts
        self._override_end_ts = end_ts

    def _global_entropy(self) -> float:
        """Compute entropy from the base linear schedule."""
        frac = min(1.0, self.num_timesteps / max(1, self.total))
        return self.ent_start + (self.ent_end - self.ent_start) * frac

    def _on_step(self) -> bool:
        global_ent = self._global_entropy()

        if (self._override_ent is not None and
                self._override_start_ts is not None):
            # Check if override has expired
            if (self._override_end_ts is not None and
                    self.num_timesteps >= self._override_end_ts):
                self._override_ent = None
                self._override_start_ts = None
                self._override_end_ts = None
                self.model.ent_coef = global_ent
            else:
                # Interpolate from override value to global value
                if self._override_end_ts is not None:
                    duration = self._override_end_ts - self._override_start_ts
                    elapsed = self.num_timesteps - self._override_start_ts
                    t = min(1.0, elapsed / max(1, duration))
                else:
                    t = 0.0
                self.model.ent_coef = self._override_ent + (global_ent - self._override_ent) * t
        else:
            self.model.ent_coef = global_ent
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


class BestVecNormalizeCallback(BaseCallback):
    """Saves VecNormalize stats alongside best_model.zip.

    Pass as callback_on_new_best to EvalCallback — fires exactly when
    a new best mean reward is found, no polling needed.
    """

    def __init__(self, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        os.makedirs(self.save_path, exist_ok=True)
        vecnorm = self.training_env
        while not isinstance(vecnorm, VecNormalize):
            vecnorm = vecnorm.venv
        stem = getattr(self.parent, "_last_best_stem", "best_model")
        vecnorm_path = os.path.join(self.save_path, f"{stem}_vecnorm.pkl")
        vecnorm.save(vecnorm_path)
        # Persist best threshold so a resumed run doesn't clobber best_model
        # on its very first evaluation (parent.best_mean_reward starts at -inf).
        score = {"best_mean_reward": float(self.parent.best_mean_reward)}
        with open(os.path.join(self.save_path, "best_score.json"), "w") as f:
            json.dump(score, f)
        if self.verbose:
            print(f"[BestVecNorm] Saved {vecnorm_path} @ step {self.num_timesteps:,}")
        return True


class VecNormalizeCheckpointCallback(BaseCallback):
    """Saves model + VecNormalize stats together at a fixed step interval."""

    def __init__(self, save_freq: int, save_path: str, name_prefix: str,
                 verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            os.makedirs(self.save_path, exist_ok=True)
            stem = f"{self.name_prefix}_{self.num_timesteps}_steps"
            model_path = os.path.join(self.save_path, stem + ".zip")
            self.model.save(model_path)
            vecnorm = self.training_env
            while not isinstance(vecnorm, VecNormalize):
                vecnorm = vecnorm.venv
            vecnorm_path = os.path.join(self.save_path, f"{stem}_vecnorm.pkl")
            vecnorm.save(vecnorm_path)
            if self.verbose:
                print(f"[Checkpoint] Saved {model_path}.zip + {vecnorm_path}")
        return True


class BestMeanEvalCallback(EvalCallback):
    """EvalCallback that saves a new best_model_{mean_reward} whenever mean reward improves."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way"
                    ) from e

            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length = np.mean(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, "
                      f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                self.logger.record("eval/success_rate", success_rate)

            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print(f"New best mean reward: {mean_reward:.2f} (prev {self.best_mean_reward:.2f})")
                self.best_mean_reward = float(mean_reward)
                if self.best_model_save_path is not None:
                    stem = f"best_model_{mean_reward:.2f}"
                    self._last_best_stem = stem
                    self.model.save(os.path.join(self.best_model_save_path, stem + ".zip"))
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training


class DeathCreditCallback(BaseCallback):
    """Redistributes death penalty across preceding frames for better credit assignment.

    The game_over penalty is a single large negative reward on the death frame,
    but the positioning mistakes that caused death happened 50-200 frames earlier.
    This callback distributes an additional linearly-increasing penalty across
    the K frames before each death event, then re-computes GAE.
    """

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        buf = self.model.rollout_buffer
        rewards = buf.rewards            # shape: (n_steps, n_envs)
        starts = buf.episode_starts      # shape: (n_steps, n_envs)
        n_steps, n_envs = rewards.shape

        modified = False
        for env_idx in range(n_envs):
            for t in range(n_steps):
                # Detect death: episode boundary with large negative reward
                is_terminal = False
                if t < n_steps - 1 and starts[t + 1, env_idx] == 1.0:
                    is_terminal = True
                elif t == n_steps - 1:
                    # Check if the buffer-final step was a death via _last_episode_starts
                    if hasattr(self.model, '_last_episode_starts'):
                        is_terminal = bool(self.model._last_episode_starts[env_idx])

                if not is_terminal:
                    continue
                if rewards[t, env_idx] >= -5.0:
                    continue  # Not a death (game_over base is -8.0)

                # Distribute penalty across preceding K frames
                for k in range(1, DEATH_CREDIT_K + 1):
                    back_t = t - k
                    if back_t < 0:
                        break
                    # Don't cross episode boundaries
                    if starts[back_t + 1, env_idx] == 1.0:
                        break
                    # Linear increase toward death: frame closest to death gets max penalty
                    penalty = DEATH_CREDIT_PER_STEP * (DEATH_CREDIT_K - k + 1) / DEATH_CREDIT_K
                    rewards[back_t, env_idx] -= penalty
                    modified = True

        if modified:
            # Re-compute GAE with modified rewards
            with torch.no_grad():
                last_values = self.model.policy.predict_values(
                    obs_as_tensor(self.model._last_obs, self.model.device))
            buf.compute_returns_and_advantage(
                last_values=last_values,
                dones=self.model._last_episode_starts)


class SelfImitationCallback(BaseCallback):
    """Collects successful hard-level trajectories and replays as BC auxiliary targets.

    Maintains a ring buffer of (obs, action) pairs from training episodes that
    clear levels >= SIL_MIN_LEVEL. After each rollout, runs BC gradient steps
    on the buffer to amplify rare positive signal from hard levels.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        # Ring buffer (allocated on first use)
        self._obs_buf = None
        self._act_buf = None
        self._buf_pos = 0
        self._buf_full = False
        self._total_collected = 0
        # Per-env episode tracking
        self._ep_obs = {}     # env_idx -> list of (obs,)
        self._ep_acts = {}    # env_idx -> list of (action,)
        self._ep_level = {}   # env_idx -> highest level seen
        self._optimizer = None

    def _on_training_start(self):
        policy = self.model.policy
        self._params = (
            list(policy.features_extractor.parameters()) +
            list(policy.mlp_extractor.policy_net.parameters()) +
            list(policy.action_net.parameters())
        )
        self._optimizer = torch.optim.Adam(self._params, lr=SIL_LR)

    def _add_to_buffer(self, obs, act):
        if self._obs_buf is None:
            obs_dim = obs.shape[-1]
            act_dim = act.shape[-1] if act.ndim > 0 else 1
            self._obs_buf = np.zeros((SIL_BUFFER_SIZE, obs_dim), dtype=np.float32)
            self._act_buf = np.zeros((SIL_BUFFER_SIZE, act_dim), dtype=np.int64)
        self._obs_buf[self._buf_pos] = obs
        self._act_buf[self._buf_pos] = act
        self._buf_pos = (self._buf_pos + 1) % SIL_BUFFER_SIZE
        if self._buf_pos == 0:
            self._buf_full = True
        self._total_collected += 1

    def _on_step(self) -> bool:
        # Track per-env episodes from self.locals
        new_obs = self.locals.get("new_obs")
        actions = self.locals.get("actions")
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        if new_obs is None or actions is None:
            return True

        n_envs = len(dones)
        for env_idx in range(n_envs):
            # Initialize tracking for new envs
            if env_idx not in self._ep_obs:
                self._ep_obs[env_idx] = []
                self._ep_acts[env_idx] = []
                self._ep_level[env_idx] = 0

            # Record current step — use pre-step obs (the state the action was
            # chosen from), not new_obs which is the post-step observation.
            self._ep_obs[env_idx].append(self.model._last_obs[env_idx].copy())
            self._ep_acts[env_idx].append(actions[env_idx].copy())

            # Track highest level
            info = infos[env_idx]
            hl = info.get("highest_level", 0)
            if hl > self._ep_level[env_idx]:
                self._ep_level[env_idx] = hl

            # Episode ended
            if dones[env_idx]:
                if self._ep_level[env_idx] >= SIL_MIN_LEVEL:
                    # Save this episode's trajectory
                    for obs, act in zip(self._ep_obs[env_idx], self._ep_acts[env_idx]):
                        self._add_to_buffer(obs, act)
                # Reset tracking
                self._ep_obs[env_idx] = []
                self._ep_acts[env_idx] = []
                self._ep_level[env_idx] = 0

        return True

    def _on_rollout_end(self):
        n_available = SIL_BUFFER_SIZE if self._buf_full else self._buf_pos
        if n_available < SIL_ACTIVATE_AT or self._optimizer is None:
            return

        policy = self.model.policy
        was_training = policy.training
        policy.train()

        device = self.model.device
        batch_size = min(256, n_available)
        total_loss = 0.0
        n_steps = 4

        for _ in range(n_steps):
            idx = np.random.choice(n_available, size=batch_size, replace=True)
            obs_b = torch.tensor(self._obs_buf[idx], dtype=torch.float32, device=device)
            act_b = torch.tensor(self._act_buf[idx], dtype=torch.int64, device=device)

            features = policy.extract_features(obs_b)
            latent_pi = policy.mlp_extractor.forward_actor(features)
            logits = policy.action_net(latent_pi)

            move_logits = logits[:, :3]
            shoot_logits = logits[:, 3:]

            bc_loss = SIL_BC_WEIGHT * (
                F.cross_entropy(move_logits, act_b[:, 0]) +
                F.cross_entropy(shoot_logits, act_b[:, 1])
            )

            self._optimizer.zero_grad()
            bc_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._params, max_norm=1.0)
            self._optimizer.step()
            total_loss += bc_loss.item()

        if not was_training:
            policy.eval()

        self.logger.record("sil/loss", total_loss / n_steps)
        self.logger.record("sil/buffer_size", n_available)
        self.logger.record("sil/total_collected", self._total_collected)


def _categorical_kl(p_logits, q_logits):
    """KL(P || Q) for categorical distributions from logits."""
    p = F.softmax(p_logits, dim=-1)
    log_p = F.log_softmax(p_logits, dim=-1)
    log_q = F.log_softmax(q_logits, dim=-1)
    return (p * (log_p - log_q)).sum(dim=-1).mean()


class PolicyAnchorCallback(BaseCallback):
    """KL-divergence anchoring to the pretrained (BC) policy.

    Snapshots the policy at training start and applies a decaying KL penalty
    on rollout observations to prevent catastrophic drift. Unlike DAPG, this
    operates on the RL observation distribution — no demo data needed, no
    distribution mismatch.

    KL(π_current || π_BC) is computed on mini-batches sampled from the current
    PPO rollout buffer. The 3× weight on shoot_kl matches DAPG's convention
    (shoot timing is rare and high-value).
    """

    def __init__(self, kl_coef_start=0.5, total_timesteps=500_000_000,
                 n_steps=8, batch_size=512, lr_ratio=0.5, verbose=0):
        super().__init__(verbose)
        self.kl_coef_start = kl_coef_start
        self.total_timesteps = total_timesteps
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.lr_ratio = lr_ratio

    def _on_training_start(self):
        # Frozen snapshot of the current (BC-pretrained) policy
        self._bc_policy = copy.deepcopy(self.model.policy)
        for p in self._bc_policy.parameters():
            p.requires_grad = False
        self._bc_policy.eval()

        # Actor-only optimizer; LR tracks PPO LR × lr_ratio
        policy = self.model.policy
        self._params = (
            list(policy.features_extractor.parameters()) +
            list(policy.mlp_extractor.policy_net.parameters()) +
            list(policy.action_net.parameters())
        )
        self._optimizer = torch.optim.Adam(self._params, lr=1e-4)

        if self.verbose:
            n_params = sum(p.numel() for p in self._bc_policy.parameters())
            print(f"[PolicyAnchor] Frozen BC snapshot ({n_params:,} params), "
                  f"kl_coef={self.kl_coef_start}, lr_ratio={self.lr_ratio}")

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        progress = min(1.0, self.num_timesteps / max(1, self.total_timesteps))
        kl_coef = self.kl_coef_start * (1.0 - progress)
        if kl_coef < 1e-6:
            return

        # Track PPO LR
        ppo_lr = self.model.policy.optimizer.param_groups[0]["lr"]
        current_lr = ppo_lr * self.lr_ratio
        for pg in self._optimizer.param_groups:
            pg["lr"] = current_lr

        # Sample from rollout buffer (filled, pre-training)
        raw_obs = self.model.rollout_buffer.observations
        obs_flat = raw_obs.reshape(-1, raw_obs.shape[-1])

        policy = self.model.policy
        was_training = policy.training
        policy.train()

        total_kl = 0.0
        for _ in range(self.n_steps):
            idx = np.random.choice(len(obs_flat),
                                   size=min(self.batch_size, len(obs_flat)),
                                   replace=True)
            obs_b = torch.tensor(obs_flat[idx], dtype=torch.float32,
                                 device=self.model.device)

            # Current policy logits
            features = policy.extract_features(obs_b)
            latent = policy.mlp_extractor.forward_actor(features)
            logits = policy.action_net(latent)

            # Frozen BC policy logits
            with torch.no_grad():
                bc_features = self._bc_policy.extract_features(obs_b)
                bc_latent = self._bc_policy.mlp_extractor.forward_actor(bc_features)
                bc_logits = self._bc_policy.action_net(bc_latent)

            # KL(current || BC) for MultiDiscrete([3, 2])
            move_kl = _categorical_kl(logits[:, :3], bc_logits[:, :3])
            shoot_kl = _categorical_kl(logits[:, 3:], bc_logits[:, 3:])
            kl_loss = kl_coef * (move_kl + 3.0 * shoot_kl)

            self._optimizer.zero_grad()
            kl_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._params, max_norm=1.0)
            self._optimizer.step()
            total_kl += kl_loss.item()

        if not was_training:
            policy.eval()

        avg_kl = total_kl / self.n_steps
        self.logger.record("kl_anchor/loss", avg_kl)
        self.logger.record("kl_anchor/coef", kl_coef)
        self.logger.record("kl_anchor/lr", current_lr)


class BCAuxiliaryCallback(BaseCallback):
    """DAPG-style BC regularization during PPO fine-tuning.

    After each rollout, does n_dapg_steps BC gradient steps (matching
    n_epochs so DAPG scales with PPO update count). Uses a separate
    optimizer whose LR tracks current PPO LR × bc_lr_ratio, so DAPG
    decays proportionally as training progresses.

    Three improvements over naive DAPG:
    - Fix A: n_dapg_steps per rollout (default: model's n_epochs) instead
      of 1. LR scales with current PPO LR × bc_lr_ratio (default 0.1).
    - Fix B: Level-stratified sampling — only demo frames from the current
      curriculum level range, so BC signal matches RL distribution.
    - Fix F: Filters powerup-active demo frames when curriculum disables
      powerups, preventing observation distribution mismatch.

    Only for MaskablePPO (feedforward policy). RecurrentPPO has an LSTM
    that requires hidden states not available in demo data.
    """

    def __init__(self, demo_paths, curriculum=None, bc_weight_start=1.0,
                 total_timesteps=500_000_000, bc_lr_ratio=0.1,
                 n_dapg_steps=None, batch_size=256, verbose=0):
        super().__init__(verbose)
        self.demo_paths = demo_paths
        self.curriculum = sorted(curriculum, key=lambda x: x[0]) if curriculum else None
        self.bc_weight_start = bc_weight_start
        self.total_timesteps = total_timesteps
        self.bc_lr_ratio = bc_lr_ratio
        self.n_dapg_steps = n_dapg_steps  # None → use model's n_epochs
        self.batch_size = batch_size
        self._bc_optimizer = None

    def _get_curriculum_phase(self):
        """Return (min_level, max_level, powerups_enabled) for current timestep."""
        if self.curriculum is None:
            return None, None, True
        phase = self.curriculum[0]
        for entry in self.curriculum:
            if self.num_timesteps >= entry[0]:
                phase = entry
        _, min_lvl, max_lvl, powerups = phase
        return min_lvl, max_lvl, powerups

    def _on_training_start(self):
        # Load and concatenate all demo files
        all_obs, all_act, all_lvl = [], [], []
        for pattern in self.demo_paths:
            for path in sorted(glob.glob(pattern)):
                data = np.load(path)
                all_obs.append(data["observations"])
                all_act.append(data["actions"])
                if "level_ids" in data:
                    all_lvl.append(data["level_ids"])
                else:
                    all_lvl.append(np.ones(len(data["actions"]), dtype=np.int32))
        if not all_obs:
            print("[BCAux] WARNING: No demo files found — BC auxiliary loss disabled")
            self._disabled = True
            return
        self._disabled = False
        obs = np.concatenate(all_obs, axis=0).astype(np.float32)
        act = np.concatenate(all_act, axis=0).astype(np.int64)
        level_ids = np.concatenate(all_lvl, axis=0).astype(np.int32)

        device = self.model.device
        self._demo_obs = torch.tensor(obs, device=device)
        self._demo_move = torch.tensor(act[:, 0], device=device)
        self._demo_shoot = torch.tensor(act[:, 1], device=device)
        self._demo_level_ids = level_ids  # keep on CPU for masking
        self._n_demos = len(obs)

        # Precompute powerup-active mask (obs indices 915=has_grid, 917=powerup_visible)
        # Values >0 mean the feature is active (both use 1.0/-1.0 encoding)
        self._demo_has_powerup = (obs[:, 915] > 0) | (obs[:, 917] > 0)

        # Compute shoot class weights
        shoot_counts = np.bincount(act[:, 1], minlength=2)
        self._shoot_weight = torch.tensor(
            [shoot_counts[1] / max(shoot_counts[0], 1), 1.0],
            dtype=torch.float32, device=device)

        # Separate optimizer for BC updates (actor-only); LR updated dynamically
        policy = self.model.policy
        self._bc_params = (
            list(policy.features_extractor.parameters()) +
            list(policy.mlp_extractor.policy_net.parameters()) +
            list(policy.action_net.parameters())
        )
        self._bc_optimizer = torch.optim.Adam(self._bc_params, lr=self.bc_lr_ratio)

        if self.verbose:
            lvl_counts = np.bincount(level_ids, minlength=14)[1:]
            print(f"[BCAux] Loaded {self._n_demos:,} demo steps, "
                  f"shoot_weight={self._shoot_weight[0].item():.1f}x, "
                  f"bc_lr_ratio={self.bc_lr_ratio}")
            print(f"[BCAux] Demo distribution by level: "
                  + " ".join(f"L{i+1}:{c}" for i, c in enumerate(lvl_counts) if c > 0))

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        if getattr(self, "_disabled", False):
            return

        # Linear decay: bc_weight_start → 0 over total_timesteps
        progress = min(1.0, self.num_timesteps / max(1, self.total_timesteps))
        bc_weight = self.bc_weight_start * (1.0 - progress)

        if bc_weight < 1e-6:
            return

        # Fix A: Update bc optimizer LR to track current PPO LR × bc_lr_ratio
        ppo_lr = self.model.policy.optimizer.param_groups[0]["lr"]
        current_bc_lr = ppo_lr * self.bc_lr_ratio
        for pg in self._bc_optimizer.param_groups:
            pg["lr"] = current_bc_lr

        # Fix B + F: Build valid-frame mask for current curriculum phase
        min_lvl, max_lvl, powerups_enabled = self._get_curriculum_phase()
        mask = np.ones(self._n_demos, dtype=bool)
        if min_lvl is not None:
            mask &= (self._demo_level_ids >= min_lvl) & (self._demo_level_ids <= max_lvl)
        if not powerups_enabled:
            mask &= ~self._demo_has_powerup
        valid_indices = np.where(mask)[0]
        if len(valid_indices) < self.batch_size // 4:
            # Fallback: level filter only (drop powerup filter if too few frames)
            if min_lvl is not None:
                level_mask = (self._demo_level_ids >= min_lvl) & (self._demo_level_ids <= max_lvl)
                valid_indices = np.where(level_mask)[0]
            if len(valid_indices) == 0:
                valid_indices = np.arange(self._n_demos)

        # Fix A: Do n_dapg_steps gradient steps (matches PPO's n_epochs)
        n_steps = self.n_dapg_steps
        if n_steps is None:
            n_steps = getattr(self.model, "n_epochs", 8)

        policy = self.model.policy
        was_training = policy.training
        policy.train()

        total_loss = 0.0
        move_logits = None
        shoot_logits = None
        shoot_b = None

        for _ in range(n_steps):
            idx = np.random.choice(valid_indices,
                                   size=min(self.batch_size, len(valid_indices)),
                                   replace=True)
            obs_b = self._demo_obs[idx]
            move_b = self._demo_move[idx]
            shoot_b = self._demo_shoot[idx]

            features = policy.extract_features(obs_b)
            latent_pi = policy.mlp_extractor.forward_actor(features)
            logits = policy.action_net(latent_pi)

            move_logits = logits[:, :3]
            shoot_logits = logits[:, 3:]

            bc_loss = bc_weight * (
                F.cross_entropy(move_logits, move_b) +
                F.cross_entropy(shoot_logits, shoot_b, weight=self._shoot_weight)
            )

            self._bc_optimizer.zero_grad()
            bc_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._bc_params, max_norm=1.0)
            self._bc_optimizer.step()
            total_loss += bc_loss.item()

        if not was_training:
            policy.eval()

        # Log metrics (from last iteration)
        avg_loss = total_loss / n_steps
        self.logger.record("bc_aux/loss", avg_loss)
        self.logger.record("bc_aux/weight", bc_weight)
        self.logger.record("bc_aux/lr", current_bc_lr)
        self.logger.record("bc_aux/n_steps", n_steps)
        self.logger.record("bc_aux/valid_demos", len(valid_indices))
        if shoot_b is not None and shoot_logits is not None:
            self.logger.record(
                "bc_aux/shoot_recall",
                (shoot_logits.detach().argmax(1)[shoot_b == 0] == 0).float().mean().item()
                if (shoot_b == 0).any() else 0.0)
