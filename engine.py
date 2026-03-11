"""Headless Bubble Trouble physics engine — no pygame dependency.

All game state is stored in numpy arrays for vectorized updates.
Designed for maximum throughput in RL training.
"""

import numpy as np
from config import (
    DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FPS,
    AGENT_WIDTH_RATIO, AGENT_HEIGHT_RATIO, AGENT_SPEED_DIVISOR,
    LASER_WIDTH_RATIO, BALL_XSPEED_DIVISOR, MAX_BALLS,
    POWERUP_NONE, POWERUP_DOUBLE_HARPOON, POWERUP_FORCE_FIELD, POWERUP_HOURGLASS,
    POWERUP_DROP_CHANCE, HOURGLASS_DURATION_SECONDS, HOURGLASS_SPEED_FACTOR,
    LEVEL_DEFS, NUM_LEVELS, DEFAULT_MAX_STEPS, REWARDS,
    compute_ball_properties,
)

# Module-level reward constants for fast access (avoid dict lookup in hot path)
REWARDS_TIME_PENALTY = REWARDS["time_penalty"]
REWARDS_WASTED_SHOT = REWARDS["wasted_shot"]
REWARDS_HIT_BALL_BASE = REWARDS["hit_ball_base"]
REWARDS_POP_BALL = REWARDS["pop_ball"]
REWARDS_DANGER_SPLIT = REWARDS["danger_split_penalty"]
REWARDS_FINISH_LEVEL = REWARDS["finish_level"]
REWARDS_TIME_BONUS_SCALE = REWARDS["time_bonus_scale"]
REWARDS_GAME_OVER = REWARDS["game_over"]
REWARDS_PICKUP = REWARDS["pickup_powerup"]
REWARDS_CLEAR_ALL = REWARDS["clear_all_levels"]

# Actions
ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_SHOOT = 2
ACTION_STILL = 3


class BubbleTroubleEngine:
    """Pure-math Bubble Trouble engine with numpy-vectorized physics."""

    def __init__(self, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, fps=DEFAULT_FPS,
                 max_steps=DEFAULT_MAX_STEPS, enable_powerups=True,
                 sequential_levels=True, start_level=1, max_level=NUM_LEVELS,
                 rng_seed=None):
        self.width = width
        self.height = height
        self.fps = fps
        self.dt = 1.0 / fps
        self.max_steps = max_steps
        self.enable_powerups = enable_powerups
        self.sequential_levels = sequential_levels
        self.start_level = start_level
        self.max_level = max_level

        # Agent dimensions
        self.agent_w = round(AGENT_WIDTH_RATIO * width)
        self.agent_h = round(AGENT_HEIGHT_RATIO * width)
        self.agent_speed = width / fps / AGENT_SPEED_DIVISOR

        # Laser properties
        self.laser_w = max(1, int(LASER_WIDTH_RATIO * width))
        self.laser_speed = height / fps

        # Precompute ball properties for each level
        self._ball_props = {}
        for lvl in range(1, 5):
            radius, max_yspeed, yacc = compute_ball_properties(lvl, width, height)
            self._ball_props[lvl] = (radius, max_yspeed, yacc)

        # Fixed normalizers for observation space (level-4 = theoretical max)
        self.max_possible_yspeed = self._ball_props[4][1]
        self.max_possible_radius = self._ball_props[4][0]

        self.rng = np.random.default_rng(rng_seed)

        # Ball state arrays (preallocated)
        self.ball_x = np.zeros(MAX_BALLS, dtype=np.float64)
        self.ball_y = np.zeros(MAX_BALLS, dtype=np.float64)
        self.ball_xspeed = np.zeros(MAX_BALLS, dtype=np.float64)
        self.ball_yspeed = np.zeros(MAX_BALLS, dtype=np.float64)
        self.ball_radius = np.zeros(MAX_BALLS, dtype=np.float64)
        self.ball_max_yspeed = np.zeros(MAX_BALLS, dtype=np.float64)
        self.ball_yacc = np.zeros(MAX_BALLS, dtype=np.float64)
        self.ball_level = np.zeros(MAX_BALLS, dtype=np.int32)
        self.n_balls = 0

        # Agent state
        self.agent_x = 0.0  # left edge of agent rect

        # Laser state (supports up to 2 for double harpoon)
        self.laser_x = np.zeros(2, dtype=np.float64)
        self.laser_length = np.zeros(2, dtype=np.float64)
        self.laser_active = np.zeros(2, dtype=bool)
        self.laser_hit_ball = np.zeros(2, dtype=bool)  # track if laser hit before deactivating
        self.max_lasers = 1  # 1 normally, 2 with double harpoon

        # Power-up state
        self.has_double_harpoon = False
        self.has_force_field = False
        self.hourglass_active = False
        self.hourglass_timer = 0.0
        # Dropped power-up on ground
        self.powerup_on_ground = False
        self.powerup_ground_type = POWERUP_NONE
        self.powerup_ground_x = 0.0
        self.powerup_ground_y = 0.0

        # Game state
        self.current_level = start_level
        self.steps = 0
        self.done = False
        self.level_cleared = False

        # Episode tracking counters
        self.shots_fired = 0
        self.shots_wasted = 0
        self.danger_splits = 0
        self.levels_cleared = 0
        self.highest_level = start_level

    def reset(self):
        """Reset the engine for a new episode. Returns None (no obs here)."""
        self.steps = 0
        self.done = False
        self.level_cleared = False
        self.current_level = self.start_level

        # Reset agent to center
        self.agent_x = (self.width - self.agent_w) / 2.0

        # Reset lasers
        self.laser_active[:] = False
        self.laser_length[:] = 0.0
        self.laser_hit_ball[:] = False
        self.max_lasers = 1

        # Reset tracking counters
        self.shots_fired = 0
        self.shots_wasted = 0
        self.danger_splits = 0
        self.levels_cleared = 0
        self.highest_level = self.start_level

        # Reset power-ups
        self.has_double_harpoon = False
        self.has_force_field = False
        self.hourglass_active = False
        self.hourglass_timer = 0.0
        self.powerup_on_ground = False
        self.powerup_ground_type = POWERUP_NONE

        # Load level
        self._load_level(self.current_level)

    def _load_level(self, level_num):
        """Load balls for a given level number."""
        self.n_balls = 0
        self.steps = 0  # Reset step counter per level

        if self.sequential_levels and level_num in LEVEL_DEFS:
            defn = LEVEL_DEFS[level_num]
            for ball_lvl, x_ratio, y_ratio in defn:
                if ball_lvl < 1 or ball_lvl > 4:
                    continue
                x = x_ratio * self.width
                y = y_ratio * self.height
                self._add_ball(ball_lvl, x, y, go_right=self.rng.random() > 0.5)
        else:
            self._load_random_level()

    def _load_random_level(self):
        """Generate a random level with ball weight up to 20."""
        total_weight = 0
        weights = {1: 1, 2: 3, 3: 7, 4: 15}
        max_ball_level = 4
        # Limit ball levels based on max_level config for curriculum
        if self.max_level <= 2:
            max_ball_level = 2
        elif self.max_level <= 4:
            max_ball_level = 3

        while total_weight < 20 and self.n_balls < MAX_BALLS:
            lvl = self.rng.integers(1, max_ball_level + 1)
            x = self.rng.uniform(0, self.width)
            y = self.rng.uniform(0, self.height * 0.5)
            self._add_ball(lvl, x, y, go_right=self.rng.random() > 0.5)
            total_weight += weights[lvl]

    def _add_ball(self, level, x, y, go_right=True, yspeed=0.0):
        """Add a ball to the active arrays."""
        if self.n_balls >= MAX_BALLS:
            return
        i = self.n_balls
        radius, max_yspeed, yacc = self._ball_props[level]
        self.ball_x[i] = x
        self.ball_y[i] = y
        self.ball_xspeed[i] = (self.width / BALL_XSPEED_DIVISOR) * (1.0 if go_right else -1.0)
        self.ball_yspeed[i] = yspeed
        self.ball_radius[i] = radius
        self.ball_max_yspeed[i] = max_yspeed
        self.ball_yacc[i] = yacc
        self.ball_level[i] = level
        self.n_balls += 1

    def _finalize_info(self, info):
        """Stamp episode-level cumulative counters onto the info dict."""
        info["shots_wasted"] = self.shots_wasted
        info["danger_splits"] = self.danger_splits
        info["levels_cleared"] = self.levels_cleared
        info["highest_level"] = self.highest_level
        return info

    def step(self, action):
        """Advance the engine by one frame. Returns (reward, terminated, truncated, info)."""
        reward = 0.0
        terminated = False
        truncated = False
        info = {"balls_hit": 0, "balls_popped": 0, "level_cleared": False,
                "game_cleared": False, "powerup_picked": False,
                "time_bonus": 0.0}

        if self.done:
            return reward, True, False, self._finalize_info(info)

        self.steps += 1

        # --- Update agent ---
        if action == ACTION_LEFT:
            self.agent_x = max(0.0, self.agent_x - self.agent_speed)
        elif action == ACTION_RIGHT:
            self.agent_x = min(self.width - self.agent_w, self.agent_x + self.agent_speed)
        elif action == ACTION_SHOOT:
            self._try_fire_laser()

        # --- Update lasers (returns wasted shot penalty) ---
        reward += self._update_lasers()

        # --- Update hourglass timer ---
        if self.hourglass_active:
            self.hourglass_timer -= self.dt
            if self.hourglass_timer <= 0:
                self.hourglass_active = False
                self.hourglass_timer = 0.0

        # --- Update balls (vectorized) ---
        n = self.n_balls
        if n > 0:
            speed_factor = HOURGLASS_SPEED_FACTOR if self.hourglass_active else 1.0
            dt = self.dt

            # Horizontal movement
            self.ball_x[:n] += self.ball_xspeed[:n] * dt * speed_factor

            # Wall bounce (left)
            left_mask = self.ball_x[:n] < 0
            self.ball_xspeed[:n][left_mask] = np.abs(self.ball_xspeed[:n][left_mask])
            self.ball_x[:n][left_mask] = -self.ball_x[:n][left_mask]

            # Wall bounce (right) — right edge is ball_x + 2*radius
            right_edge = self.ball_x[:n] + 2 * self.ball_radius[:n]
            right_mask = right_edge > self.width
            self.ball_xspeed[:n][right_mask] = -np.abs(self.ball_xspeed[:n][right_mask])
            overshoot = right_edge[right_mask] - self.width
            self.ball_x[:n][right_mask] -= overshoot

            # Vertical movement (kinematic: y += vy*dt + 0.5*a*dt^2)
            self.ball_y[:n] += (self.ball_yspeed[:n] * dt * speed_factor
                                + 0.5 * self.ball_yacc[:n] * (dt * speed_factor) ** 2)

            # Floor bounce — bottom edge is ball_y + 2*radius
            bottom_edge = self.ball_y[:n] + 2 * self.ball_radius[:n]
            floor_mask = bottom_edge > self.height
            self.ball_yspeed[:n][floor_mask] = -self.ball_max_yspeed[:n][floor_mask]
            self.ball_y[:n][floor_mask] = self.height - 2 * self.ball_radius[:n][floor_mask]

            # Update vertical speed (gravity)
            self.ball_yspeed[:n] += self.ball_yacc[:n] * dt * speed_factor
            # Clamp vertical speed
            self.ball_yspeed[:n] = np.clip(
                self.ball_yspeed[:n],
                -self.ball_max_yspeed[:n],
                self.ball_max_yspeed[:n]
            )

            # Ceiling check — balls that hit ceiling get popped/split
            ceiling_mask = self.ball_y[:n] < 0
            if np.any(ceiling_mask):
                indices = np.where(ceiling_mask)[0]
                for idx in sorted(indices, reverse=True):
                    lvl = self.ball_level[idx]
                    bx = self.ball_x[idx]
                    if lvl == 1:
                        self._remove_ball(idx)
                    else:
                        self._remove_ball(idx)
                        self._add_ball(lvl - 1, bx, 0.0, go_right=False)
                        self._add_ball(lvl - 1, bx, 0.0, go_right=True)
                n = self.n_balls

        # --- Check laser-ball collisions ---
        hit_reward, hit_info = self._check_laser_ball_collisions()
        reward += hit_reward
        info["balls_hit"] += hit_info["balls_hit"]
        info["balls_popped"] += hit_info["balls_popped"]

        # --- Check agent-ball collisions ---
        if self._check_agent_ball_collision():
            if self.has_force_field:
                self.has_force_field = False
            else:
                reward += REWARDS_GAME_OVER
                terminated = True
                self.done = True
                return reward, terminated, truncated, self._finalize_info(info)

        # --- Check power-up pickup ---
        if self.powerup_on_ground and self.enable_powerups:
            if self._check_powerup_pickup():
                reward += REWARDS_PICKUP
                info["powerup_picked"] = True

        # --- Check level cleared ---
        if self.n_balls == 0:
            reward += REWARDS_FINISH_LEVEL
            # Time bonus: reward faster clears
            time_bonus = REWARDS_TIME_BONUS_SCALE * (1.0 - self.steps / self.max_steps)
            reward += time_bonus
            info["level_cleared"] = True
            info["time_bonus"] = time_bonus
            self.levels_cleared += 1

            if self.sequential_levels and self.current_level < self.max_level:
                self.current_level += 1
                self.highest_level = max(self.highest_level, self.current_level)
                self._load_level(self.current_level)
                # Reset lasers and ground power-ups for new level
                self.laser_active[:] = False
                self.laser_length[:] = 0.0
                self.powerup_on_ground = False
                self.powerup_ground_type = POWERUP_NONE
            else:
                # All levels cleared or non-sequential mode
                if self.sequential_levels:
                    reward += REWARDS_CLEAR_ALL
                    info["game_cleared"] = True
                truncated = True
                self.done = True
                return reward, terminated, truncated, self._finalize_info(info)

        # --- Time penalty and time limit ---
        reward += REWARDS_TIME_PENALTY

        if self.steps >= self.max_steps:
            truncated = True
            self.done = True

        return reward, terminated, truncated, self._finalize_info(info)

    def _try_fire_laser(self):
        """Fire a laser if a slot is available."""
        agent_center_x = self.agent_x + self.agent_w / 2.0
        for i in range(self.max_lasers):
            if not self.laser_active[i]:
                self.laser_active[i] = True
                self.laser_x[i] = agent_center_x
                self.laser_length[i] = self.agent_h  # Start from agent top
                self.laser_hit_ball[i] = False
                self.shots_fired += 1
                return

    def _update_lasers(self):
        """Grow active lasers upward. Returns wasted shot penalty."""
        penalty = 0.0
        for i in range(self.max_lasers):
            if self.laser_active[i]:
                self.laser_length[i] = min(self.laser_length[i] + self.laser_speed, self.height)
                if self.laser_length[i] >= self.height:
                    self.laser_active[i] = False
                    self.laser_length[i] = 0.0
                    if not self.laser_hit_ball[i]:
                        penalty += REWARDS_WASTED_SHOT
                        self.shots_wasted += 1
                    self.laser_hit_ball[i] = False
        return penalty

    def _check_laser_ball_collisions(self):
        """Check all lasers against all balls. Returns (reward, info_dict)."""
        reward = 0.0
        hit_info = {"balls_hit": 0, "balls_popped": 0, "danger_splits": 0}
        n = self.n_balls
        if n == 0:
            return reward, hit_info

        agent_center_x = self.agent_x + self.agent_w / 2.0
        danger_x_threshold = 2.0 * self.agent_w
        danger_y_threshold = 0.7 * self.height

        for laser_idx in range(self.max_lasers):
            if not self.laser_active[laser_idx]:
                continue

            lx = self.laser_x[laser_idx]
            laser_top = self.height - self.laser_length[laser_idx]

            # Ball centers and radii
            bcx = self.ball_x[:n] + self.ball_radius[:n]  # center x
            bcy = self.ball_y[:n] + self.ball_radius[:n]   # center y
            br = self.ball_radius[:n]

            # Check overlap: ball x-range includes laser x, ball y-range overlaps laser y-range
            x_overlap = (bcx - br <= lx) & (bcx + br >= lx)
            y_overlap = (bcy + br >= laser_top) & (bcy - br <= self.height)
            hits = x_overlap & y_overlap

            if not np.any(hits):
                continue

            # Process first hit only (laser deactivates on hit)
            hit_idx = np.where(hits)[0][0]
            lvl = self.ball_level[hit_idx]
            bx = self.ball_x[hit_idx]
            by = self.ball_y[hit_idx]
            b_radius = float(self.ball_radius[hit_idx])  # capture before swap-with-last removal

            # Mark laser as having hit a ball
            self.laser_hit_ball[laser_idx] = True

            # Deactivate laser
            self.laser_active[laser_idx] = False
            self.laser_length[laser_idx] = 0.0

            if lvl == 1:
                # Smallest ball — just remove
                reward += self._pop_ball_at(hit_idx)
                hit_info["balls_popped"] += 1
            else:
                # Split: remove this ball, add 2 smaller ones
                reward += self._hit_ball_at(hit_idx, lvl)
                hit_info["balls_hit"] += 1

                # Danger split check: ball center close to agent horizontally
                # AND ball bottom edge is below 70% of screen height (ball is low)
                ball_bottom = by + 2 * b_radius
                ball_cx = bx + b_radius
                if (abs(ball_cx - agent_center_x) < danger_x_threshold
                        and ball_bottom > danger_y_threshold):
                    reward += REWARDS_DANGER_SPLIT
                    hit_info["danger_splits"] += 1
                    self.danger_splits += 1

                self._add_ball(lvl - 1, bx, by, go_right=False)
                self._add_ball(lvl - 1, bx, by, go_right=True)

            # Update n since we modified ball arrays
            n = self.n_balls
            break  # One hit per laser per frame

        return reward, hit_info

    def _pop_ball_at(self, idx):
        """Remove ball at idx (smallest ball, fully destroyed). Returns reward."""
        self._maybe_drop_powerup(self.ball_x[idx], self.ball_y[idx])
        self._remove_ball(idx)
        return REWARDS_POP_BALL

    def _hit_ball_at(self, idx, level):
        """Remove ball at idx (will split). Returns level-scaled reward."""
        self._maybe_drop_powerup(self.ball_x[idx], self.ball_y[idx])
        self._remove_ball(idx)
        return REWARDS_HIT_BALL_BASE * level

    def _remove_ball(self, idx):
        """Remove ball at index by swapping with last active ball."""
        last = self.n_balls - 1
        if idx != last:
            self.ball_x[idx] = self.ball_x[last]
            self.ball_y[idx] = self.ball_y[last]
            self.ball_xspeed[idx] = self.ball_xspeed[last]
            self.ball_yspeed[idx] = self.ball_yspeed[last]
            self.ball_radius[idx] = self.ball_radius[last]
            self.ball_max_yspeed[idx] = self.ball_max_yspeed[last]
            self.ball_yacc[idx] = self.ball_yacc[last]
            self.ball_level[idx] = self.ball_level[last]
        self.n_balls -= 1

    def _maybe_drop_powerup(self, x, y):
        """Possibly drop a power-up at the given position."""
        if not self.enable_powerups or self.powerup_on_ground:
            return
        if self.rng.random() < POWERUP_DROP_CHANCE:
            self.powerup_on_ground = True
            self.powerup_ground_x = x
            self.powerup_ground_y = y
            self.powerup_ground_type = self.rng.integers(1, 4)  # 1, 2, or 3

    def _check_powerup_pickup(self):
        """Check if agent overlaps with ground power-up. Returns True if picked up."""
        if not self.powerup_on_ground:
            return False

        # Simple overlap: power-up is a point, agent is a rect
        agent_left = self.agent_x
        agent_right = self.agent_x + self.agent_w
        agent_top = self.height - self.agent_h
        agent_bottom = self.height

        px, py = self.powerup_ground_x, self.powerup_ground_y
        if agent_left <= px <= agent_right and agent_top <= py <= agent_bottom:
            self._apply_powerup(self.powerup_ground_type)
            self.powerup_on_ground = False
            self.powerup_ground_type = POWERUP_NONE
            return True
        return False

    def _apply_powerup(self, ptype):
        """Apply a power-up effect."""
        if ptype == POWERUP_DOUBLE_HARPOON:
            self.has_double_harpoon = True
            self.max_lasers = 2
        elif ptype == POWERUP_FORCE_FIELD:
            self.has_force_field = True
        elif ptype == POWERUP_HOURGLASS:
            self.hourglass_active = True
            self.hourglass_timer = HOURGLASS_DURATION_SECONDS

    def _check_agent_ball_collision(self):
        """Check if any ball overlaps with agent rect. Returns True if collision."""
        n = self.n_balls
        if n == 0:
            return False

        # Agent rect
        ax_left = self.agent_x
        ax_right = self.agent_x + self.agent_w
        ay_top = self.height - self.agent_h
        ay_bottom = self.height

        # Ball centers and radii
        bcx = self.ball_x[:n] + self.ball_radius[:n]
        bcy = self.ball_y[:n] + self.ball_radius[:n]
        br = self.ball_radius[:n]

        # Find closest point on agent rect to each ball center
        closest_x = np.clip(bcx, ax_left, ax_right)
        closest_y = np.clip(bcy, ay_top, ay_bottom)

        # Distance from closest point to ball center
        dx = bcx - closest_x
        dy = bcy - closest_y
        dist_sq = dx * dx + dy * dy

        # Collision if distance < radius
        return np.any(dist_sq < br * br)

    def get_state(self):
        """Return a snapshot of the current game state for observation/rendering."""
        return {
            "agent_x": self.agent_x,
            "agent_w": self.agent_w,
            "agent_h": self.agent_h,
            "laser_x": self.laser_x[:self.max_lasers].copy(),
            "laser_length": self.laser_length[:self.max_lasers].copy(),
            "laser_active": self.laser_active[:self.max_lasers].copy(),
            "ball_x": self.ball_x[:self.n_balls].copy(),
            "ball_y": self.ball_y[:self.n_balls].copy(),
            "ball_xspeed": self.ball_xspeed[:self.n_balls].copy(),
            "ball_yspeed": self.ball_yspeed[:self.n_balls].copy(),
            "ball_radius": self.ball_radius[:self.n_balls].copy(),
            "ball_level": self.ball_level[:self.n_balls].copy(),
            "n_balls": self.n_balls,
            "current_level": self.current_level,
            "steps": self.steps,
            "max_steps": self.max_steps,
            "width": self.width,
            "height": self.height,
            "has_double_harpoon": self.has_double_harpoon,
            "has_force_field": self.has_force_field,
            "hourglass_active": self.hourglass_active,
            "hourglass_timer": self.hourglass_timer,
            "powerup_on_ground": self.powerup_on_ground,
            "powerup_ground_x": self.powerup_ground_x,
            "powerup_ground_y": self.powerup_ground_y,
            "powerup_ground_type": self.powerup_ground_type,
        }
