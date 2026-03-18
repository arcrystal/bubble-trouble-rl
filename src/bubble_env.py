"""Gymnasium environment wrapping the headless Bubble Trouble engine."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from engine import BubbleTroubleEngine
from config import (
    DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FPS,
    MAX_OBS_BALLS, OBS_PER_BALL, OBS_SIZE, MAX_BALLS, MAX_OBSTACLES, DEFAULT_MAX_STEPS,
    NUM_LEVELS, MAX_BALL_LEVEL,
    OBSTACLE_DOOR, OBSTACLE_OPENING, OBSTACLE_LOWERING_CEIL, MAX_OBSTACLE_TYPE,
    POP_BASE_IMPULSE_RATIO, POP_VELOCITY_INHERIT, POP_CHAIN_GAIN, POP_MASS_EXPONENT,
)


class BubbleTroubleEnv(gym.Env):
    """Bubble Trouble as a Gymnasium environment.

    Observation: 224-element float32 vector (see config.py for layout).
    Actions: MultiDiscrete([3, 2]) — move (LEFT/RIGHT/STILL) × shoot (SHOOT/NO_SHOOT).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": DEFAULT_FPS}

    def __init__(self, render_mode=None, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
                 fps=DEFAULT_FPS, max_steps=DEFAULT_MAX_STEPS,
                 enable_powerups=True, sequential_levels=True,
                 start_level=1, max_level=NUM_LEVELS):
        super().__init__()

        self.render_mode = render_mode
        self.width = width
        self.height = height

        self.engine = BubbleTroubleEngine(
            width=width, height=height, fps=fps, max_steps=max_steps,
            enable_powerups=enable_powerups, sequential_levels=sequential_levels,
            start_level=start_level, max_level=max_level,
        )

        # MultiDiscrete: [move(3), shoot(2)] — allows simultaneous move + shoot
        self.action_space = spaces.MultiDiscrete([3, 2])
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )

        self._renderer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.engine.rng = np.random.default_rng(seed)
        self.engine.reset()
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        reward, terminated, truncated, engine_info = self.engine.step(action)
        obs = self._get_obs()
        info = self._get_info()
        info.update(engine_info)
        return obs, reward, terminated, truncated, info

    def action_masks(self):
        """Return valid action masks for MaskablePPO.

        For MultiDiscrete([3, 2]), returns a flat bool array of shape (5,):
        [LEFT, RIGHT, STILL, SHOOT, NO_SHOOT]
        - move (0-2): always all valid
        - shoot (3-4): SHOOT masked if no laser slot available
        """
        e = self.engine
        can_fire = any(not e.laser_active[i] for i in range(e.max_lasers))
        return np.array([True, True, True, can_fire, True])

    def _get_obs(self):
        """Build the 224-element observation vector."""
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        e = self.engine
        n = e.n_balls
        width = e.width
        height = e.height
        agent_cx = e.agent_x + e.agent_w / 2.0

        # --- Ball features (sorted by distance to agent center) ---
        if n > 0:
            ball_cx = e.ball_x[:n] + e.ball_radius[:n]

            distances = np.abs(ball_cx - agent_cx)
            sorted_idx = np.argsort(distances)

            k = min(n, MAX_OBS_BALLS)
            idx = sorted_idx[:k]

            # Normalized ball features (9 per ball)
            base = 0
            obs[base:base + k] = (e.ball_x[idx] + e.ball_radius[idx]) / width * 2 - 1  # x: [-1, 1]
            base = MAX_OBS_BALLS
            obs[base:base + k] = (e.ball_y[idx] + e.ball_radius[idx]) / height * 2 - 1  # y: [-1, 1]
            base = MAX_OBS_BALLS * 2
            max_xspeed = width / 9.4
            obs[base:base + k] = e.ball_xspeed[idx] / max_xspeed  # xspeed: ~[-1, 1]
            base = MAX_OBS_BALLS * 3
            obs[base:base + k] = e.ball_yspeed[idx] / e.max_possible_yspeed  # yspeed: ~[-1, 1], fixed normalizer
            base = MAX_OBS_BALLS * 4
            obs[base:base + k] = e.ball_radius[idx] / e.max_possible_radius * 2 - 1  # radius: [-1, 1]
            base = MAX_OBS_BALLS * 5
            obs[base:base + k] = e.ball_level[idx] / MAX_BALL_LEVEL * 2 - 1  # level: [-1, 1]
            base = MAX_OBS_BALLS * 6
            obs[base:base + k] = 1.0  # is_active flag

            # Relative x: signed horizontal distance from agent to ball center
            base = MAX_OBS_BALLS * 7
            rel_x = (e.ball_x[idx] + e.ball_radius[idx] - agent_cx) / (width / 2)
            obs[base:base + k] = np.clip(rel_x, -1.0, 1.0)

            # Peak height: predicted apex y when ball is rising/at peak, 1.0 when falling
            base = MAX_OBS_BALLS * 8
            rising = e.ball_yspeed[idx] <= 0
            ball_cy = e.ball_y[idx] + e.ball_radius[idx]
            peak_cy = np.where(
                rising,
                ball_cy - e.ball_yspeed[idx] ** 2 / (2 * e.ball_yacc[idx]),
                float(e.effective_height),  # falling → peak is "at floor"
            )
            obs[base:base + k] = np.clip(peak_cy / e.effective_height * 2 - 1, -1.0, 1.0)

            # Intercept x: predicted ball center x when laser reaches ball's current height.
            # The laser travels screen_height pixels per second, so it takes ball_cy/height
            # seconds to reach a ball at height ball_cy. In that time the ball moves
            # ball_xspeed * (ball_cy / height) pixels horizontally.
            # Directly encodes "where to stand to intercept this ball" — a non-linear
            # product of position, speed, and the laser speed constant that an MLP cannot
            # easily derive from raw per-ball features without explicit knowledge of laser speed.
            base = MAX_OBS_BALLS * 9
            t_reach_s = ball_cy / height          # seconds for laser to reach ball's current y
            intercept_cx = (e.ball_x[idx] + e.ball_radius[idx]) + e.ball_xspeed[idx] * t_reach_s
            obs[base:base + k] = np.clip(intercept_cx / width * 2 - 1, -1.0, 1.0)

        # --- Agent features (5) ---
        agent_base = MAX_OBS_BALLS * OBS_PER_BALL
        obs[agent_base] = agent_cx / width * 2 - 1  # agent x: [-1, 1]

        any_laser_active = np.any(e.laser_active[:e.max_lasers])
        obs[agent_base + 1] = 1.0 if any_laser_active else -1.0  # laser active

        max_laser_len = 0.0
        best_laser_idx = -1
        for i in range(e.max_lasers):
            if e.laser_active[i] and e.laser_length[i] > max_laser_len:
                max_laser_len = e.laser_length[i]
                best_laser_idx = i
        obs[agent_base + 2] = max_laser_len / height * 2 - 1  # laser length: [-1, 1]

        # Laser x-position: where the active laser is horizontally
        if best_laser_idx >= 0:
            obs[agent_base + 3] = e.laser_x[best_laser_idx] / width * 2 - 1
        else:
            obs[agent_base + 3] = agent_cx / width * 2 - 1  # continuity: laser would fire from here

        # Can-fire: can the agent fire a new laser right now?
        can_fire = any(not e.laser_active[i] for i in range(e.max_lasers))
        obs[agent_base + 4] = 1.0 if can_fire else -1.0

        # --- Global features (5) ---
        global_base = agent_base + 5
        obs[global_base] = n / MAX_BALLS * 2 - 1  # ball count ratio: [-1, 1]
        remaining = max(0, e.max_steps - e.steps)
        obs[global_base + 1] = remaining / e.max_steps * 2 - 1  # time remaining: [-1, 1]
        obs[global_base + 2] = e.current_level / NUM_LEVELS * 2 - 1  # current level: auto-scales with NUM_LEVELS

        # Chain ceiling targeting: pre-computes "if I hit ball X now, will its children
        # reach the ceiling?" using the full pop physics model (velocity inheritance,
        # chain-depth impulse, level-dependent gravity, kinematic prediction).
        # Returns the intercept_x and quality of the best chain opportunity on screen.
        # Critical for levels 9-10 (lvl-5/6 balls) where ceiling pop chains are the
        # only viable strategy. An MLP cannot derive this — it requires knowing
        # POP_VELOCITY_INHERIT, POP_BASE_IMPULSE_RATIO, POP_CHAIN_GAIN, per-level yacc,
        # and solving a kinematic equation.
        best_chain_quality = 0.0
        best_chain_intercept = agent_cx  # default: agent's current position
        pop_base = POP_BASE_IMPULSE_RATIO * height
        if n > 0:
            for bi in range(n):
                lvl_i = int(e.ball_level[bi])
                if lvl_i <= 1:
                    continue  # lvl-1 balls don't split
                child_level = lvl_i - 1
                child_yacc = e._ball_props[child_level][2]
                parent_ys = float(e.ball_yspeed[bi])
                ball_cy_i = float(e.ball_y[bi] + e.ball_radius[bi])
                if ball_cy_i <= 0:
                    continue  # already at ceiling
                chain_d = int(e.ball_chain_depth[bi]) + 1
                chain_f = (1.0 + POP_CHAIN_GAIN) ** chain_d
                impulse = pop_base * chain_f / (child_level ** POP_MASS_EXPONENT)
                child_ys = POP_VELOCITY_INHERIT * parent_ys - impulse  # negative = upward
                if child_ys >= 0:
                    continue  # children won't move upward
                max_rise = child_ys ** 2 / (2.0 * child_yacc)
                quality = max_rise / ball_cy_i  # > 1 = children reach ceiling
                # Weight by ball level (higher level = more value in ceiling-popping)
                weighted_q = quality * lvl_i / MAX_BALL_LEVEL
                if weighted_q > best_chain_quality:
                    best_chain_quality = weighted_q
                    # Intercept x for this ball
                    t_r = ball_cy_i / height
                    best_chain_intercept = float(e.ball_x[bi] + e.ball_radius[bi]) + float(e.ball_xspeed[bi]) * t_r
        obs[global_base + 3] = np.clip(best_chain_intercept / width * 2 - 1, -1.0, 1.0)
        obs[global_base + 4] = np.clip(best_chain_quality / 2.0 * 2 - 1, -1.0, 1.0)  # 0→-1, 1→0, 2→+1

        # --- Power-up features (6) ---
        pu_base = global_base + 5
        obs[pu_base] = 1.0 if e.has_laser_grid else -1.0
        obs[pu_base + 1] = 1.0 if np.any(e.laser_stuck[:e.max_lasers]) else -1.0
        powerup_visible = e.powerup_on_ground or e.powerup_falling
        obs[pu_base + 2] = 1.0 if powerup_visible else -1.0
        if powerup_visible:
            dist = (e.powerup_ground_x - agent_cx) / width * 2  # signed distance
            obs[pu_base + 3] = np.clip(dist, -1.0, 1.0)
        else:
            obs[pu_base + 3] = 0.0
        # n_rising_ratio: fraction of active balls currently rising (yspeed < 0).
        # Aggregates 16 individual yspeed features into one policy-relevant scalar:
        # high value = many balls in upward phase = wide ceiling-pop window open now.
        if n > 0:
            n_rising = int(np.sum(e.ball_yspeed[:n] < 0))
            obs[pu_base + 4] = n_rising / n * 2.0 - 1.0   # [-1=all falling, +1=all rising]
        else:
            obs[pu_base + 4] = -1.0

        # closest_approach_time: seconds until the nearest approaching ball is at agent x.
        # For the closest ball moving toward the agent, t = |relative_x| / |xspeed|.
        # This is a division across two feature channels — non-trivial for an MLP —
        # and directly tells the agent how urgently it must dodge.
        # Normalized: -1.0 = ball overhead now (0s), +1.0 = 3+ seconds away or no threat.
        if n > 0:
            ball_cx_all = e.ball_x[:n] + e.ball_radius[:n]
            rel_x_all = ball_cx_all - agent_cx
            # Approaching = ball moving toward agent (rel_x and xspeed have opposite signs)
            approaching = (rel_x_all * e.ball_xspeed[:n]) < 0
            if np.any(approaching):
                ap_idx = np.where(approaching)[0]
                times = np.abs(rel_x_all[ap_idx]) / (np.abs(e.ball_xspeed[ap_idx]) + 1e-6)
                min_t = float(np.min(times))
                obs[pu_base + 5] = np.clip(min_t / 3.0, 0.0, 1.0) * 2.0 - 1.0
            else:
                obs[pu_base + 5] = 1.0   # no balls approaching — safe
        else:
            obs[pu_base + 5] = 1.0

        # --- Obstacle features (MAX_OBSTACLES * 6 = 48) ---
        # Per obstacle: center_x, center_y, width, height, type, is_passable
        # is_passable: 1.0 = agent can currently walk through, -1.0 = blocked
        #   STATIC:         always -1.0
        #   DOOR:           1.0 when open (timer < 0), else -1.0
        #   OPENING:        1.0 when animating/sliding (timer < 0), else -1.0
        #   LOWERING_CEIL:  always 1.0 (never blocks agent horizontally)
        obs_base = pu_base + 6
        no = e.n_obstacles
        if no > 0:
            k = min(no, MAX_OBSTACLES)
            for oi in range(k):
                b = obs_base + oi * 6
                otype = int(e.obs_type[oi])
                obs[b]     = (e.obs_x[oi] + e.obs_w[oi] / 2) / width * 2 - 1    # center x
                obs[b + 1] = (e.obs_y[oi] + e.obs_h[oi] / 2) / height * 2 - 1   # center y
                obs[b + 2] = e.obs_w[oi] / width * 2 - 1                          # width
                obs[b + 3] = e.obs_h[oi] / height * 2 - 1                         # height
                obs[b + 4] = otype / MAX_OBSTACLE_TYPE * 2 - 1                    # type: [-1, 1]
                if otype == OBSTACLE_DOOR or otype == OBSTACLE_OPENING:
                    obs[b + 5] = 1.0 if e.obs_timer[oi] < 0 else -1.0
                elif otype == OBSTACLE_LOWERING_CEIL:
                    obs[b + 5] = 1.0   # never blocks agent horizontally
                else:
                    obs[b + 5] = -1.0  # STATIC: always blocks

        return obs

    def _get_info(self):
        e = self.engine
        return {
            "current_level": e.current_level,
            "n_balls": e.n_balls,
            "steps": e.steps,
        }

    def render(self):
        if self.render_mode is None:
            return None

        if self._renderer is None:
            from renderer import PygameRenderer
            self._renderer = PygameRenderer(self.width, self.height)

        state = self.engine.get_state()
        if self.render_mode == "human":
            self._renderer.render(state)
        elif self.render_mode == "rgb_array":
            return self._renderer.render_to_array(state)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def set_curriculum(self, start_level=None, max_level=None, enable_powerups=None):
        """Update curriculum parameters between episodes."""
        if start_level is not None:
            self.engine.start_level = start_level
        if max_level is not None:
            self.engine.max_level = max_level
        if enable_powerups is not None:
            self.engine.enable_powerups = enable_powerups
