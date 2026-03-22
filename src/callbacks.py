"""Shared training callbacks for Bubble Trouble RL."""

import os

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize


class CurriculumCallback(BaseCallback):
    """Advances curriculum phases based on timestep schedule."""

    def __init__(self, curriculum, eval_callback=None, verbose: int = 0):
        super().__init__(verbose)
        self.curriculum = sorted(curriculum, key=lambda x: x[0])
        self.current_phase = 0
        self.eval_callback = eval_callback

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

        _, min_l, max_l, _ = self.curriculum[self.current_phase]
        self.logger.record("train/curriculum_phase", f"{self.current_phase}: {min_l}-{max_l}")
        if self.eval_callback and hasattr(self.eval_callback, "best_mean_reward"):
            self.logger.record("eval/best_mean_reward", self.eval_callback.best_mean_reward)
        return True


class EntropyScheduleCallback(BaseCallback):
    """Linearly decays ent_coef from start to end over total_timesteps.

    Uses absolute num_timesteps so resuming continues the decay correctly.
    """

    def __init__(self, start: float, end: float, total_timesteps: int,
                 verbose: int = 0):
        super().__init__(verbose)
        self.ent_start = start
        self.ent_end = end
        self.total = total_timesteps

    def _on_step(self) -> bool:
        frac = min(1.0, self.num_timesteps / max(1, self.total))
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


class BestVecNormalizeCallback(BaseCallback):
    """Saves VecNormalize stats whenever EvalCallback finds a new best model.

    Must be ordered AFTER EvalCallback in CallbackList so best_mean_reward
    is already updated when checked.
    """

    def __init__(self, eval_callback, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.eval_callback = eval_callback
        self.save_path = save_path
        self._last_best = float("-inf")

    def _on_step(self) -> bool:
        best = getattr(self.eval_callback, "best_mean_reward", float("-inf"))
        if best > self._last_best:
            self._last_best = best
            os.makedirs(self.save_path, exist_ok=True)
            vecnorm = self.training_env
            while not isinstance(vecnorm, VecNormalize):
                vecnorm = vecnorm.venv
            vecnorm_path = os.path.join(self.save_path, "best_model_vecnorm.pkl")
            vecnorm.save(vecnorm_path)
            if self.verbose:
                print(f"[BestVecNorm] Saved {vecnorm_path} "
                      f"(best_mean_reward={best:.2f} @ step {self.num_timesteps:,})")
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
            model_path = os.path.join(self.save_path, stem)
            self.model.save(model_path)
            vecnorm = self.training_env
            while not isinstance(vecnorm, VecNormalize):
                vecnorm = vecnorm.venv
            vecnorm_path = os.path.join(self.save_path, f"{stem}_vecnorm.pkl")
            vecnorm.save(vecnorm_path)
            if self.verbose:
                print(f"[Checkpoint] Saved {model_path}.zip + {vecnorm_path}")
        return True
