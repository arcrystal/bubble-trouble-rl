"""Gymnasium environment wrapping the headless Bubble Trouble engine."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from engine import BubbleTroubleEngine
from config import (
    DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FPS,
    MAX_OBS_BALLS, OBS_SIZE, MAX_BALLS, DEFAULT_MAX_STEPS,
    NUM_LEVELS, HOURGLASS_DURATION_SECONDS,
)


class BubbleTroubleEnv(gym.Env):
    """Bubble Trouble as a Gymnasium environment.

    Observation: 110-element float32 vector (see config.py for layout).
    Actions: 0=LEFT, 1=RIGHT, 2=SHOOT, 3=STILL.
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

        self.action_space = spaces.Discrete(4)
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

    def _get_obs(self):
        """Build the 110-element observation vector."""
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        e = self.engine
        n = e.n_balls
        agent_cx = e.agent_x + e.agent_w / 2.0

        # --- Ball features (sorted by distance to agent center) ---
        if n > 0:
            ball_cx = e.ball_x[:n] + e.ball_radius[:n]

            distances = np.abs(ball_cx - agent_cx)
            sorted_idx = np.argsort(distances)

            k = min(n, MAX_OBS_BALLS)
            idx = sorted_idx[:k]

            # Normalized ball features
            base = 0
            obs[base:base + k] = (e.ball_x[idx] + e.ball_radius[idx]) / e.width * 2 - 1  # x: [-1, 1]
            base = MAX_OBS_BALLS
            obs[base:base + k] = (e.ball_y[idx] + e.ball_radius[idx]) / e.height * 2 - 1  # y: [-1, 1]
            base = MAX_OBS_BALLS * 2
            max_xspeed = e.width / 9.4
            obs[base:base + k] = e.ball_xspeed[idx] / max_xspeed  # xspeed: ~[-1, 1]
            base = MAX_OBS_BALLS * 3
            obs[base:base + k] = e.ball_yspeed[idx] / e.max_possible_yspeed  # yspeed: ~[-1, 1], fixed normalizer
            base = MAX_OBS_BALLS * 4
            obs[base:base + k] = e.ball_radius[idx] / e.max_possible_radius * 2 - 1  # radius: [-1, 1]
            base = MAX_OBS_BALLS * 5
            obs[base:base + k] = 1.0  # is_active flag

        # --- Agent features (5) ---
        agent_base = MAX_OBS_BALLS * 6
        obs[agent_base] = agent_cx / e.width * 2 - 1  # agent x: [-1, 1]

        any_laser_active = np.any(e.laser_active[:e.max_lasers])
        obs[agent_base + 1] = 1.0 if any_laser_active else -1.0  # laser active

        max_laser_len = 0.0
        best_laser_idx = -1
        for i in range(e.max_lasers):
            if e.laser_active[i] and e.laser_length[i] > max_laser_len:
                max_laser_len = e.laser_length[i]
                best_laser_idx = i
        obs[agent_base + 2] = max_laser_len / e.height * 2 - 1  # laser length: [-1, 1]

        # Laser x-position: where the active laser is horizontally
        if best_laser_idx >= 0:
            obs[agent_base + 3] = e.laser_x[best_laser_idx] / e.width * 2 - 1
        else:
            obs[agent_base + 3] = agent_cx / e.width * 2 - 1  # continuity: laser would fire from here

        # Can-fire: can the agent fire a new laser right now?
        can_fire = any(not e.laser_active[i] for i in range(e.max_lasers))
        obs[agent_base + 4] = 1.0 if can_fire else -1.0

        # --- Global features (3) ---
        global_base = agent_base + 5
        obs[global_base] = n / MAX_BALLS * 2 - 1  # ball count ratio: [-1, 1]
        remaining = max(0, e.max_steps - e.steps)
        obs[global_base + 1] = remaining / e.max_steps * 2 - 1  # time remaining: [-1, 1]
        obs[global_base + 2] = e.current_level / NUM_LEVELS * 2 - 1  # current level: [-0.71, 1.0]

        # --- Power-up features (6) ---
        pu_base = global_base + 3
        obs[pu_base] = 1.0 if e.has_double_harpoon else -1.0
        obs[pu_base + 1] = 1.0 if e.has_force_field else -1.0
        obs[pu_base + 2] = 1.0 if e.hourglass_active else -1.0
        obs[pu_base + 3] = (e.hourglass_timer / HOURGLASS_DURATION_SECONDS * 2 - 1) if e.hourglass_active else -1.0
        obs[pu_base + 4] = 1.0 if e.powerup_on_ground else -1.0
        if e.powerup_on_ground:
            dist = (e.powerup_ground_x - agent_cx) / e.width * 2  # signed distance
            obs[pu_base + 5] = np.clip(dist, -1.0, 1.0)
        else:
            obs[pu_base + 5] = 0.0

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
