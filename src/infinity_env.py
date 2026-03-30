"""Gymnasium environment for Bubble Trouble infinity mode.

Reward: full base engine reward function (time_penalty, hit/pop, ceiling pop, game_over, etc.)
— same reward structure as regular mode for model interchangeability.
Observation: same 968-element format as BubbleTroubleEnv (unified obs space).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from infinity_engine import InfinityEngine
from config import (
    DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FPS,
    MAX_OBS_BALLS, OBS_PER_BALL, OBS_AGENT, OBS_GLOBAL, OBS_POWERUP,
    OBS_SIZE, MAX_BALLS, MAX_OBSTACLES,
    MAX_BALL_LEVEL, BALL_XSPEED_DIVISOR,
    OBSTACLE_DOOR, OBSTACLE_OPENING, OBSTACLE_LOWERING_CEIL, MAX_OBSTACLE_TYPE,
    POP_BASE_IMPULSE_RATIO, POP_VELOCITY_INHERIT, POP_CHAIN_GAIN, POP_MASS_EXPONENT,
    BALL_FLAG_STATIC, POWERUP_LASER_GRID, POWERUP_HOURGLASS,
    REWARDS,
)
from infinity_config import SPAWN_SCHEDULE


class InfinityModeEnv(gym.Env):
    """Bubble Trouble infinity mode as a Gymnasium environment.

    Uses the same 968-dim observation space as BubbleTroubleEnv so that
    models are fully interchangeable between regular and infinity modes.
    Only the reward function differs.

    Actions: MultiDiscrete([3, 2]) -- move (LEFT/RIGHT/STILL) x shoot (SHOOT/NO_SHOOT).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": DEFAULT_FPS}

    def __init__(self, seed=None, render_mode=None, render_speed=1.0):
        super().__init__()
        self.render_mode = render_mode
        self.render_speed = render_speed
        self.width = DEFAULT_WIDTH
        self.height = DEFAULT_HEIGHT

        self.engine = InfinityEngine(
            width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
            fps=DEFAULT_FPS, seed=seed,
        )

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(OBS_SIZE,),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([3, 2])
        self._renderer = None
        self._obs_buf = np.zeros(OBS_SIZE, dtype=np.float32)  # pre-allocated obs buffer

    def action_masks(self):
        """Return valid action masks for MaskablePPO.

        For MultiDiscrete([3, 2]), returns flat bool array of shape (5,):
        [LEFT, RIGHT, STILL, SHOOT, NO_SHOOT]
        """
        e = self.engine
        can_fire = any(not e.laser_active[i] for i in range(e.max_lasers))
        return np.array([True, True, True, can_fire, True])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.engine._fixed_seed = seed
        self.engine.reset()
        return self._get_obs(), self._get_info()

    def step(self, action):
        reward, terminated, truncated, engine_info = self.engine.step(action)
        info = self._get_info()
        info.update(engine_info)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_info(self):
        e = self.engine
        return {
            "n_balls": e.n_balls,
            "elapsed_time": e.elapsed_time,
            "difficulty": e._difficulty(),
            "schedule_idx": e._schedule_idx,
        }

    # ------------------------------------------------------------------
    # Observation — same 968-dim format as BubbleTroubleEnv
    # ------------------------------------------------------------------

    def _get_obs(self):
        """Build OBS_SIZE (968) observation vector, identical layout to BubbleTroubleEnv."""
        obs = self._obs_buf
        obs[:] = 0.0
        e = self.engine
        n = e.n_balls
        width = e.width
        height = e.height
        eff_h = e.effective_height
        agent_cx = e.agent_x + e.agent_w / 2.0
        NB = MAX_OBS_BALLS  # 64

        # --- Ball features (64 x 14, feature-major) ---
        if n > 0:
            # If more balls than slots, select the 64 closest to agent
            if n > NB:
                all_cx = e.ball_x[:n] + e.ball_radius[:n]
                dists = np.abs(all_cx - agent_cx)
                idx = np.argpartition(dists, NB)[:NB]
                k = NB
            else:
                idx = np.arange(n)
                k = n

            ball_cx = e.ball_x[idx] + e.ball_radius[idx]
            ball_cy = e.ball_y[idx] + e.ball_radius[idx]

            base = 0
            obs[base:base + k] = np.clip(ball_cx / width * 2 - 1, -1.0, 1.0)            # 0: x
            base = NB
            obs[base:base + k] = np.clip(ball_cy / height * 2 - 1, -1.0, 1.0)           # 1: y
            base = NB * 2
            max_xspeed = width / BALL_XSPEED_DIVISOR
            obs[base:base + k] = np.clip(e.ball_xspeed[idx] / max_xspeed, -1.0, 1.0)   # 2: xspeed
            base = NB * 3
            obs[base:base + k] = np.clip(e.ball_yspeed[idx] / e.max_possible_yspeed, -1.0, 1.0)  # 3: yspeed
            base = NB * 4
            obs[base:base + k] = e.ball_radius[idx] / e.max_possible_radius * 2 - 1     # 4: radius
            base = NB * 5
            obs[base:base + k] = e.ball_level[idx] / MAX_BALL_LEVEL * 2 - 1             # 5: level
            base = NB * 6
            obs[base:base + k] = 1.0                                                    # 6: is_active

            base = NB * 7
            rel_x = (ball_cx - agent_cx) / (width / 2)
            obs[base:base + k] = np.clip(rel_x, -1.0, 1.0)                              # 7: relative_x

            base = NB * 8
            rising = e.ball_yspeed[idx] <= 0
            apex_rising = ball_cy - e.ball_yspeed[idx] ** 2 / (2 * e.ball_yacc[idx])
            actual_bounce_ys = e.ball_max_yspeed[idx] * np.sqrt(e.ball_bounciness[idx])
            apex_falling = eff_h - actual_bounce_ys ** 2 / (2 * e.ball_yacc[idx])
            peak_cy = np.where(rising, apex_rising, apex_falling)
            obs[base:base + k] = np.clip(peak_cy / eff_h * 2 - 1, -1.0, 1.0)            # 8: peak_height

            base = NB * 9
            t_reach_s = (eff_h - ball_cy) / height
            intercept_cx = ball_cx + e.ball_xspeed[idx] * t_reach_s
            obs[base:base + k] = np.clip(intercept_cx / width * 2 - 1, -1.0, 1.0)       # 9: intercept_x

            base = NB * 10
            obs[base:base + k] = np.clip(e.ball_bounciness[idx] / 3.0 * 2 - 1, -1.0, 1.0)  # 10: bounciness

            base = NB * 11
            obs[base:base + k] = np.clip(e.ball_chain_depth[idx] / 5.0 * 2 - 1, -1.0, 1.0)  # 11: chain_depth

            base = NB * 12
            obs[base:base + k] = np.where(
                e.ball_flags[idx] == BALL_FLAG_STATIC, 1.0, -1.0)                        # 12: is_static

            base = NB * 13
            height_factor = 1.0 + REWARDS["height_bonus_factor"] * np.maximum(0.0, 1.0 - ball_cy / eff_h)
            obs[base:base + k] = np.clip(height_factor / 1.5 * 2 - 1, -1.0, 1.0)        # 13: height_bonus

        # --- Agent features (6) ---
        agent_base = NB * OBS_PER_BALL
        obs[agent_base] = agent_cx / width * 2 - 1

        any_laser_active = np.any(e.laser_active[:e.max_lasers])
        obs[agent_base + 1] = 1.0 if any_laser_active else -1.0

        max_laser_len = 0.0
        best_laser_idx = -1
        for i in range(e.max_lasers):
            if e.laser_active[i] and e.laser_length[i] > max_laser_len:
                max_laser_len = e.laser_length[i]
                best_laser_idx = i
        obs[agent_base + 2] = max_laser_len / height * 2 - 1

        if best_laser_idx >= 0:
            obs[agent_base + 3] = e.laser_x[best_laser_idx] / width * 2 - 1
        else:
            obs[agent_base + 3] = agent_cx / width * 2 - 1

        can_fire = any(not e.laser_active[i] for i in range(e.max_lasers))
        obs[agent_base + 4] = 1.0 if can_fire else -1.0

        # Laser max reach at current position (obstacle scan)
        max_reach = float(eff_h)
        for oi in range(e.n_obstacles):
            otype = e.obs_type[oi]
            if e.obs_timer[oi] < 0 and otype in (OBSTACLE_DOOR, OBSTACLE_OPENING):
                continue  # passable
            ox, ow = e.obs_x[oi], e.obs_w[oi]
            if ox <= agent_cx <= ox + ow:
                obs_bottom = e.obs_y[oi] + e.obs_h[oi]
                reachable = eff_h - obs_bottom
                if reachable > 0:
                    max_reach = min(max_reach, reachable)
        obs[agent_base + 5] = max_reach / eff_h * 2 - 1

        # --- Global features (13) — same layout as regular mode ---
        global_base = agent_base + OBS_AGENT
        obs[global_base] = n / e.max_balls * 2 - 1                                       # ball count ratio
        # Map difficulty (schedule progress) → "time remaining" proxy
        difficulty = e._difficulty()
        obs[global_base + 1] = (1.0 - difficulty) * 2 - 1                               # time remaining proxy
        # Map schedule progress → "level" proxy (loops reset to 0)
        schedule_progress = e._schedule_idx / len(SPAWN_SCHEDULE)
        obs[global_base + 2] = schedule_progress * 2 - 1                                # level proxy

        # Chain ceiling targeting (best opportunity)
        best_chain_quality = 0.0
        best_chain_intercept = agent_cx
        pop_base = POP_BASE_IMPULSE_RATIO * height
        if n > 0:
            for bi in range(n):
                lvl_i = int(e.ball_level[bi])
                if lvl_i <= 1:
                    continue
                child_level = lvl_i - 1
                child_yacc = e._ball_props[child_level][2]
                parent_ys = float(e.ball_yspeed[bi])
                ball_cy_i = float(e.ball_y[bi] + e.ball_radius[bi])
                if ball_cy_i <= 0:
                    continue
                chain_d = int(e.ball_chain_depth[bi]) + 1
                chain_f = (1.0 + POP_CHAIN_GAIN) ** chain_d
                impulse = pop_base * chain_f / (child_level ** POP_MASS_EXPONENT)
                child_ys = POP_VELOCITY_INHERIT * parent_ys - impulse
                if child_ys >= 0:
                    continue
                max_rise = child_ys ** 2 / (2.0 * child_yacc)
                quality = max_rise / ball_cy_i
                weighted_q = quality * lvl_i / MAX_BALL_LEVEL
                if weighted_q > best_chain_quality:
                    best_chain_quality = weighted_q
                    t_r = (eff_h - ball_cy_i) / height
                    best_chain_intercept = float(e.ball_x[bi] + e.ball_radius[bi]) + float(e.ball_xspeed[bi]) * t_r
        obs[global_base + 3] = np.clip(best_chain_intercept / width * 2 - 1, -1.0, 1.0)
        obs[global_base + 4] = np.clip(best_chain_quality / 2.0 * 2 - 1, -1.0, 1.0)

        # n_rising_ratio
        if n > 0:
            n_rising = int(np.sum(e.ball_yspeed[:n] < 0))
            obs[global_base + 5] = n_rising / n * 2.0 - 1.0
        else:
            obs[global_base + 5] = -1.0

        # closest_approach_time
        if n > 0:
            ball_cx_all = e.ball_x[:n] + e.ball_radius[:n]
            rel_x_all = ball_cx_all - agent_cx
            approaching = (rel_x_all * e.ball_xspeed[:n]) < 0
            if np.any(approaching):
                ap_idx = np.where(approaching)[0]
                times = np.abs(rel_x_all[ap_idx]) / (np.abs(e.ball_xspeed[ap_idx]) + 1e-6)
                min_t = float(np.min(times))
                obs[global_base + 6] = np.clip(min_t / 3.0, 0.0, 1.0) * 2.0 - 1.0
            else:
                obs[global_base + 6] = 1.0
        else:
            obs[global_base + 6] = 1.0

        # Ball level histogram (6 values, levels 1-6)
        if n > 0:
            counts = np.bincount(e.ball_level[:n].astype(int), minlength=MAX_BALL_LEVEL + 1)
            for lvl in range(1, MAX_BALL_LEVEL + 1):
                obs[global_base + 7 + (lvl - 1)] = counts[lvl] / e.max_balls * 2 - 1
        else:
            obs[global_base + 7:global_base + 13] = -1.0

        # --- Power-up features (5) ---
        pu_base = global_base + OBS_GLOBAL
        obs[pu_base] = 1.0 if e.has_laser_grid else -1.0
        obs[pu_base + 1] = 1.0 if np.any(e.laser_stuck[:e.max_lasers]) else -1.0
        powerup_visible = e.powerup_on_ground or e.powerup_falling
        obs[pu_base + 2] = 1.0 if powerup_visible else -1.0
        if powerup_visible:
            dist = (e.powerup_ground_x - agent_cx) / width * 2
            obs[pu_base + 3] = np.clip(dist, -1.0, 1.0)
        else:
            obs[pu_base + 3] = 0.0
        if powerup_visible:
            if e.powerup_ground_type == POWERUP_LASER_GRID:
                obs[pu_base + 4] = 1.0
            elif e.powerup_ground_type == POWERUP_HOURGLASS:
                obs[pu_base + 4] = -1.0

        # --- Obstacle features (MAX_OBSTACLES * 6 = 48) ---
        obs_base = pu_base + OBS_POWERUP
        no = e.n_obstacles
        if no > 0:
            k = min(no, MAX_OBSTACLES)
            for oi in range(k):
                b = obs_base + oi * 6
                otype = int(e.obs_type[oi])
                obs[b]     = np.clip((e.obs_x[oi] + e.obs_w[oi] / 2) / width * 2 - 1, -1.0, 1.0)
                obs[b + 1] = np.clip((e.obs_y[oi] + e.obs_h[oi] / 2) / height * 2 - 1, -1.0, 1.0)
                obs[b + 2] = np.clip(e.obs_w[oi] / width * 2 - 1, -1.0, 1.0)
                obs[b + 3] = np.clip(e.obs_h[oi] / height * 2 - 1, -1.0, 1.0)
                obs[b + 4] = otype / MAX_OBSTACLE_TYPE * 2 - 1
                if otype == OBSTACLE_DOOR or otype == OBSTACLE_OPENING:
                    obs[b + 5] = 1.0 if e.obs_timer[oi] < 0 else -1.0
                elif otype == OBSTACLE_LOWERING_CEIL:
                    obs[b + 5] = 1.0
                else:
                    obs[b + 5] = -1.0

        return obs

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self):
        if self.render_mode is None:
            return None

        if self._renderer is None:
            from renderer import PygameRenderer
            self._renderer = PygameRenderer(self.width, self.height, speed=self.render_speed)

        state = self.engine.get_state()
        if self.render_mode == "human":
            self._renderer.render(state)
        elif self.render_mode == "rgb_array":
            return self._renderer.render_to_array(state)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
