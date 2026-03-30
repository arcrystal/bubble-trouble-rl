"""Infinity mode engine — schedule-based ball spawning, open arena.

Subclasses BubbleTroubleEngine, adding:
  - Deterministic spawn schedule that loops indefinitely
  - Random color, bounciness, fly-in side, and entry height per spawn
  - Probabilistic high-bounciness inheritance on splits
  - Deterministic replay via seeded RNG

The base class handles all physics, laser, and agent mechanics unchanged.
No obstacles, no scene phases — just an open arena with escalating pressure.
"""

import numpy as np
from engine import BubbleTroubleEngine
from config import (
    DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FPS,
    BALL_FLAG_NORMAL,
)
from infinity_config import (
    SPAWN_SCHEDULE,
    BALL_COLORS,
    HIGH_BOUNCE_PROB, HIGH_BOUNCE_RANGE, BOUNCE_INHERIT_PROB,
    MAX_BOUNCE_HEIGHT_RATIO,
    SPAWN_SAFETY_RATIO, MAX_BALLS,
)


class InfinityEngine(BubbleTroubleEngine):
    """Bubble Trouble engine with schedule-based continuous spawning.

    Balls spawn according to SPAWN_SCHEDULE from infinity_config.py.
    The schedule loops once exhausted, creating sustained pressure.
    """

    def __init__(self, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
                 fps=DEFAULT_FPS, seed=None):
        super().__init__(
            width=width, height=height, fps=fps,
            max_steps=10**9,             # no timeout in infinity mode
            enable_powerups=True,
            sequential_levels=False,     # we manage progression ourselves
            start_level=1, max_level=1,
            rng_seed=None,               # don't seed parent RNG
            max_balls=MAX_BALLS,
        )

        self._fixed_seed = seed
        self._enable_level_clear = False  # No level clearing in infinity mode

        # High-bounciness tracker (parallel to ball arrays)
        self.ball_high_bounce = np.zeros(MAX_BALLS, dtype=bool)

        # Timing
        self.elapsed_time = 0.0
        self.spawn_timer = 0.0

        # Schedule cursor
        self._schedule_idx = 0

        # Seeded RNG for deterministic spawning
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):
        actual_seed = (self._fixed_seed if self._fixed_seed is not None
                       else int(np.random.default_rng().integers(0, 2**31)))
        self._rng = np.random.default_rng(actual_seed)
        # Also seed the base engine's RNG for deterministic powerup drops
        self.rng = np.random.default_rng(actual_seed)

        self.elapsed_time = 0.0
        self._schedule_idx = 0
        self.ball_high_bounce[:] = False

        self.n_balls = 0  # Clear balls before base reset (our _load_level is no-op)
        self.effective_height = self.height  # No short maps in infinity
        super().reset()

        # Spawn the first ball immediately (delay=0 in schedule)
        delay, level, side = SPAWN_SCHEDULE[0]
        self.spawn_timer = delay
        if delay == 0:
            self._spawn_ball(level, side)
            self._advance_schedule()
            # Set timer to next entry's delay so the second ball isn't instant
            next_delay, _, _ = SPAWN_SCHEDULE[self._schedule_idx]
            self.spawn_timer = next_delay

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action):
        dt = 1.0 / self.fps
        self.elapsed_time += dt

        # Spawn balls according to schedule
        self.spawn_timer -= dt
        while self.spawn_timer <= 0.0:
            delay, level, side = SPAWN_SCHEDULE[self._schedule_idx]
            self._spawn_ball(level, side)
            self._advance_schedule()
            # Next entry's delay (min dt prevents zero-delay entries from
            # causing multiple spawns in a single frame on schedule wrap)
            next_delay, _, _ = SPAWN_SCHEDULE[self._schedule_idx]
            self.spawn_timer += max(next_delay, dt)

        # Run base physics (level-clear check is disabled via _enable_level_clear)
        reward, terminated, truncated, info = super().step(action)
        return reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Schedule
    # ------------------------------------------------------------------

    def _advance_schedule(self):
        self._schedule_idx = (self._schedule_idx + 1) % len(SPAWN_SCHEDULE)

    def _difficulty(self):
        """Approximate difficulty as schedule progress (0→1 over first loop)."""
        return min(1.0, self._schedule_idx / len(SPAWN_SCHEDULE))

    # ------------------------------------------------------------------
    # Ball spawning
    # ------------------------------------------------------------------

    def _cap_bounciness(self, level, bounciness):
        """Cap bounciness so the ball never bounces above MAX_BOUNCE_HEIGHT_RATIO of screen."""
        radius, max_yspeed, yacc = self._ball_props[level]
        default_bounce_h = max_yspeed ** 2 / (2.0 * yacc)
        max_bounce_h = MAX_BOUNCE_HEIGHT_RATIO * self.effective_height - 2 * radius
        if max_bounce_h <= 0:
            return 1.0
        max_b = max_bounce_h / default_bounce_h
        return min(bounciness, max_b)

    def _spawn_ball(self, level, side=None):
        """Spawn a ball. side: "L"=left, "R"=right, None=random."""
        if self.n_balls >= self.max_balls:
            return

        # Determine side
        if side == "L":
            from_left = True
        elif side == "R":
            from_left = False
        else:
            from_left = bool(self._rng.integers(2))
        spawn_x = 0.0 if from_left else float(self.width)

        # Safety: skip if agent is near spawn edge
        agent_x_ratio = self.agent_x / self.width
        edge_ratio = 0.0 if from_left else 1.0
        if abs(agent_x_ratio - edge_ratio) < SPAWN_SAFETY_RATIO:
            if side is not None:
                return  # explicit side requested, can't flip
            # Try the other side
            from_left = not from_left
            spawn_x = 0.0 if from_left else float(self.width)
            edge_ratio = 0.0 if from_left else 1.0
            if abs(agent_x_ratio - edge_ratio) < SPAWN_SAFETY_RATIO:
                return  # agent in the middle-ish, skip this spawn

        # Default peak height: ball enters at its natural bounce apex (yspeed=0)
        radius, max_yspeed, yacc = self._ball_props[level]
        bounce_h = max_yspeed ** 2 / (2.0 * yacc)
        spawn_y = self.effective_height - 2 * radius - bounce_h

        # Random color
        color_idx = int(self._rng.integers(len(BALL_COLORS)))
        color = BALL_COLORS[color_idx]

        # Bounciness (capped so bounce stays below 75% of screen)
        high_bounce = self._rng.random() < HIGH_BOUNCE_PROB
        bounciness = (float(self._rng.uniform(*HIGH_BOUNCE_RANGE))
                      if high_bounce else 1.0)
        bounciness = self._cap_bounciness(level, bounciness)
        high_bounce = high_bounce and bounciness > 1.01

        idx = self._add_ball(
            level, spawn_x, spawn_y,
            go_right=from_left,
            yspeed=0.0,
            bounciness=bounciness,
            flags=BALL_FLAG_NORMAL,
            color=color,
            chain_depth=0,
            inherit_bounciness=high_bounce,
        )
        if idx is not None:
            self.ball_high_bounce[idx] = high_bounce

    # ------------------------------------------------------------------
    # Bounciness inheritance on split
    # ------------------------------------------------------------------

    def _add_ball(self, level, x, y, go_right=True, yspeed=0.0, bounciness=1.0,
                  flags=BALL_FLAG_NORMAL, color=None, chain_depth=0,
                  inherit_bounciness=False, section=-1):
        """Override to cap bounciness and handle probabilistic inheritance on splits."""
        # Cap bounciness for this level before adding
        bounciness = self._cap_bounciness(level, bounciness)
        idx = super()._add_ball(level, x, y, go_right, yspeed, bounciness,
                                flags, color, chain_depth, inherit_bounciness, section)
        if idx is not None and chain_depth > 0 and bounciness > 1.01:
            # Child inherited high bounciness from parent via base class logic.
            # Apply probabilistic revert.
            if self._rng.random() >= BOUNCE_INHERIT_PROB:
                self.ball_bounciness[idx] = 1.0
                self.ball_inherit_bounciness[idx] = False
                self.ball_high_bounce[idx] = False
            else:
                self.ball_high_bounce[idx] = True
        return idx

    def _remove_ball(self, idx):
        """Override to keep ball_high_bounce in sync with swap-with-last."""
        last = self.n_balls - 1
        if idx != last:
            self.ball_high_bounce[idx] = self.ball_high_bounce[last]
        self.ball_high_bounce[last] = False
        super()._remove_ball(idx)

    # ------------------------------------------------------------------
    # Overrides to prevent base class level management
    # ------------------------------------------------------------------

    def _load_level(self, level_num):
        """No-op — infinity mode has no levels."""
        pass

    # ------------------------------------------------------------------
    # State snapshot for renderer / observation
    # ------------------------------------------------------------------

    def get_state(self):
        state = super().get_state()
        state["infinity_mode"] = True
        state["elapsed_time"] = self.elapsed_time
        state["difficulty"] = self._difficulty()
        state["schedule_idx"] = self._schedule_idx
        state["schedule_len"] = len(SPAWN_SCHEDULE)
        return state
