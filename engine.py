"""Headless Bubble Trouble physics engine — no pygame dependency.

All game state is stored in numpy arrays for vectorized updates.
Designed for maximum throughput in RL training.
"""

import numpy as np
from config import (
    DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FPS,
    AGENT_WIDTH_RATIO, AGENT_HEIGHT_RATIO, AGENT_SPEED_DIVISOR,
    AGENT_HITBOX_RECTS, AGENT_HITBOX_SHRINK,
    LASER_WIDTH_RATIO, BALL_XSPEED_DIVISOR, MAX_BALLS, MAX_BALL_LEVEL, MAX_OBSTACLES,
    POP_BASE_IMPULSE_RATIO, POP_VELOCITY_INHERIT, POP_CHAIN_GAIN, POP_MASS_EXPONENT,
    POWERUP_NONE, POWERUP_DOUBLE_HARPOON, POWERUP_FORCE_FIELD, POWERUP_HOURGLASS,
    POWERUP_DROP_CHANCE, HOURGLASS_DURATION_SECONDS, HOURGLASS_SPEED_FACTOR,
    LEVEL_DEFS, OBSTACLE_DEFS, NUM_LEVELS, DEFAULT_MAX_STEPS, REWARDS,
    BALL_FLAG_NORMAL, BALL_FLAG_STATIC,
    OBSTACLE_STATIC, OBSTACLE_DOOR, OBSTACLE_OPENING, OBSTACLE_LOWERING_CEIL,
    DOOR_TRIGGER_LEFT, DOOR_TRIGGER_RIGHT,
    OPENING_WALL_DELAY_S, OPENING_WALL_SPEED,
    LOWERING_CEIL_TARGET_RATIO, LOWERING_CEIL_SPEED_RATIO,
    LEVEL_HEIGHT_OVERRIDE,
    BALL_COLORS_BY_LEVEL,
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

        # Agent hitbox: 5 rects tracing the sprite silhouette (precomputed pixels)
        self._hitbox_rects = [
            (xr * self.agent_w, yr * self.agent_h,
             wr * self.agent_w, hr * self.agent_h)
            for (xr, yr, wr, hr) in AGENT_HITBOX_RECTS
        ]
        self._hitbox_shrink = AGENT_HITBOX_SHRINK * (self.width / DEFAULT_WIDTH)

        # Laser properties
        self.laser_w = max(1, int(LASER_WIDTH_RATIO * width))
        self.laser_speed = height / fps

        # Precompute ball properties for each level
        self._ball_props = {}
        for lvl in range(1, MAX_BALL_LEVEL + 1):
            radius, max_yspeed, yacc = compute_ball_properties(lvl, width, height)
            self._ball_props[lvl] = (radius, max_yspeed, yacc)

        # Pop physics: chain-depth-aware impulse
        self.pop_base_impulse = POP_BASE_IMPULSE_RATIO * height
        self.pop_velocity_inherit = POP_VELOCITY_INHERIT
        self.pop_chain_gain = POP_CHAIN_GAIN
        self.pop_mass_exponent = POP_MASS_EXPONENT

        # Fixed normalizers for observation space (max ball level = theoretical max)
        # yspeed normalizer accounts for chain pop speeds (up to ~4× normal bounce)
        self.max_possible_yspeed = self._ball_props[MAX_BALL_LEVEL][1] * 4.0
        self.max_possible_radius = self._ball_props[MAX_BALL_LEVEL][0]

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
        self.ball_bounciness = np.ones(MAX_BALLS, dtype=np.float64)
        self.ball_flags = np.zeros(MAX_BALLS, dtype=np.int32)
        self.ball_color = np.zeros((MAX_BALLS, 3), dtype=np.uint8)  # RGB color inherited from ancestor
        self.ball_chain_depth = np.zeros(MAX_BALLS, dtype=np.int32)  # consecutive pops without floor bounce
        self.ball_inherit_bounciness = np.ones(MAX_BALLS, dtype=bool)  # if False, children get bounce=1.0
        self.n_balls = 0

        # Obstacle state arrays (preallocated)
        self.obs_x = np.zeros(MAX_OBSTACLES, dtype=np.float64)
        self.obs_y = np.zeros(MAX_OBSTACLES, dtype=np.float64)
        self.obs_w = np.zeros(MAX_OBSTACLES, dtype=np.float64)
        self.obs_h = np.zeros(MAX_OBSTACLES, dtype=np.float64)
        self.obs_type = np.zeros(MAX_OBSTACLES, dtype=np.int32)
        self.obs_timer = np.zeros(MAX_OBSTACLES, dtype=np.float64)  # For opening walls / lowering ceiling
        self.n_obstacles = 0

        # Effective play area height (may be overridden per level for short maps)
        self.effective_height = height

        # Agent state
        self.agent_x = 0.0  # left edge of agent rect

        # Laser state (supports up to 2 for double harpoon)
        self.laser_x = np.zeros(2, dtype=np.float64)
        self.laser_length = np.zeros(2, dtype=np.float64)
        self.laser_max_length = np.full(2, float(height), dtype=np.float64)  # max reachable (obstacle ceiling)
        self.laser_active = np.zeros(2, dtype=bool)
        self.laser_hit_ball = np.zeros(2, dtype=bool)  # track if laser hit before deactivating
        self.max_lasers = 1  # 1 normally, 2 with double harpoon

        # Power-up state
        self.has_double_harpoon = False
        self.has_force_field = False
        self.hourglass_active = False
        self.hourglass_timer = 0.0
        # Dropped power-up (falls from pop position to floor, then pickable)
        self.powerup_on_ground = False  # True once it has landed
        self.powerup_falling = False    # True while falling
        self.powerup_ground_type = POWERUP_NONE
        self.powerup_ground_x = 0.0
        self.powerup_ground_y = 0.0
        self.powerup_fall_speed = height / fps * 0.5  # falls at half laser speed

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

        # Pop effects tracking (renderer-only, cleared each step)
        self.recent_pops = []

    def reset(self):
        """Reset the engine for a new episode. Returns None (no obs here)."""
        self.steps = 0
        self.done = False
        self.level_cleared = False
        self.current_level = self.start_level

        # Reset agent to center
        self.agent_x = (self.width - self.agent_w) / 2.0

        # Reset lasers (max_length updated after _load_level sets effective_height)
        self.laser_active[:] = False
        self.laser_length[:] = 0.0
        self.laser_hit_ball[:] = False
        self.max_lasers = 1

        # Reset pop effects
        self.recent_pops = []

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
        self.powerup_falling = False
        self.powerup_ground_type = POWERUP_NONE

        # Load level (sets effective_height)
        self._load_level(self.current_level)
        self.laser_max_length[:] = self.effective_height

        # Place agent on the correct side of door walls
        self._position_agent_for_doors()

    def _load_level(self, level_num):
        """Load balls and obstacles for a given level number."""
        self.n_balls = 0
        self.steps = 0  # Reset step counter per level

        # Set effective height (short map override)
        if level_num in LEVEL_HEIGHT_OVERRIDE:
            self.effective_height = self.height * LEVEL_HEIGHT_OVERRIDE[level_num]
        else:
            self.effective_height = self.height

        # Load obstacles for this level
        self._load_obstacles(level_num)

        if self.sequential_levels and level_num in LEVEL_DEFS:
            defn = LEVEL_DEFS[level_num]
            for ball_def in defn:
                ball_lvl = ball_def["lvl"]
                if ball_lvl < 1 or ball_lvl > MAX_BALL_LEVEL:
                    continue
                x = ball_def["x"] * self.width
                y = ball_def["y"] * self.height
                go_right        = ball_def.get("dir", "R") == "R"
                flags           = BALL_FLAG_STATIC if ball_def.get("static", False) else BALL_FLAG_NORMAL
                bounciness      = ball_def.get("bounce", 1.0)
                inherit_bounce  = ball_def.get("keep_bounce", True)
                color           = ball_def.get("color", None)  # RGB tuple or None → default
                self._add_ball(ball_lvl, x, y, go_right=go_right,
                               bounciness=bounciness, flags=flags,
                               color=color,
                               inherit_bounciness=inherit_bounce)
        else:
            self._load_random_level()

    def _load_obstacles(self, level_num):
        """Load obstacle rectangles for a level from OBSTACLE_DEFS."""
        self.n_obstacles = 0
        self.obs_timer[:] = 0.0
        if level_num not in OBSTACLE_DEFS:
            return
        defs = OBSTACLE_DEFS[level_num]
        for i, obs_def in enumerate(defs):
            if i >= MAX_OBSTACLES:
                break
            xr, yr, wr, hr = obs_def[0], obs_def[1], obs_def[2], obs_def[3]
            otype = obs_def[4] if len(obs_def) > 4 else OBSTACLE_STATIC
            self.obs_x[i] = xr * self.width
            self.obs_y[i] = yr * self.height
            self.obs_w[i] = wr * self.width
            self.obs_h[i] = hr * self.height
            self.obs_type[i] = otype
            # Initialize timer/extra data for special obstacle types
            if otype == OBSTACLE_OPENING:
                self.obs_timer[i] = OPENING_WALL_DELAY_S
            elif otype == OBSTACLE_LOWERING_CEIL:
                self.obs_timer[i] = 0.0  # starts immediately
            elif otype == OBSTACLE_DOOR:
                # Store trigger side in obs_timer (0=left, 1=right)
                trigger = obs_def[5] if len(obs_def) > 5 else DOOR_TRIGGER_LEFT
                self.obs_timer[i] = float(trigger)
            self.n_obstacles += 1

    def _load_random_level(self):
        """Generate a random level with ball weight up to 20."""
        total_weight = 0
        weights = {lvl: 2 ** lvl - 1 for lvl in range(1, MAX_BALL_LEVEL + 1)}
        max_ball_level = MAX_BALL_LEVEL
        # Limit ball levels based on max_level config for curriculum
        if self.max_level <= 3:
            max_ball_level = min(max_ball_level, 2)
        elif self.max_level <= 6:
            max_ball_level = min(max_ball_level, 3)
        elif self.max_level <= 10:
            max_ball_level = min(max_ball_level, 4)
        elif self.max_level <= 18:
            max_ball_level = min(max_ball_level, 5)

        while total_weight < 20 and self.n_balls < MAX_BALLS:
            lvl = self.rng.integers(1, max_ball_level + 1)
            x = self.rng.uniform(0, self.width)
            y = self.rng.uniform(0, self.height * 0.5)
            self._add_ball(lvl, x, y, go_right=self.rng.random() > 0.5)
            total_weight += weights[lvl]

    def _add_ball(self, level, x, y, go_right=True, yspeed=0.0, bounciness=1.0,
                  flags=BALL_FLAG_NORMAL, color=None, chain_depth=0,
                  inherit_bounciness=True):
        """Add a ball to the active arrays."""
        if self.n_balls >= MAX_BALLS:
            return
        i = self.n_balls
        radius, max_yspeed, yacc = self._ball_props[level]
        self.ball_x[i] = x
        self.ball_y[i] = y
        xspeed = (self.width / BALL_XSPEED_DIVISOR) * (1.0 if go_right else -1.0)
        self.ball_xspeed[i] = 0.0 if flags == BALL_FLAG_STATIC else xspeed
        self.ball_yspeed[i] = yspeed
        self.ball_radius[i] = radius
        self.ball_max_yspeed[i] = max_yspeed
        self.ball_yacc[i] = yacc
        self.ball_level[i] = level
        self.ball_bounciness[i] = bounciness
        self.ball_flags[i] = flags
        rgb = color if color is not None else BALL_COLORS_BY_LEVEL[level]
        self.ball_color[i] = rgb
        self.ball_chain_depth[i] = chain_depth
        self.ball_inherit_bounciness[i] = inherit_bounciness
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
        self.recent_pops = []

        # --- Update agent ---
        if action == ACTION_LEFT:
            self.agent_x = max(0.0, self.agent_x - self.agent_speed)
        elif action == ACTION_RIGHT:
            self.agent_x = min(self.width - self.agent_w, self.agent_x + self.agent_speed)
        elif action == ACTION_SHOOT:
            self._try_fire_laser()

        # --- Agent-obstacle collision (push back) ---
        self._resolve_agent_obstacles()

        # --- Update lasers (returns wasted shot penalty) ---
        reward += self._update_lasers()

        # --- Update hourglass timer ---
        if self.hourglass_active:
            self.hourglass_timer -= self.dt
            if self.hourglass_timer <= 0:
                self.hourglass_active = False
                self.hourglass_timer = 0.0

        # --- Update dynamic obstacles (opening walls, lowering ceiling) ---
        self._update_dynamic_obstacles()

        # --- Update falling power-up ---
        if self.powerup_falling:
            self.powerup_ground_y += self.powerup_fall_speed
            if self.powerup_ground_y >= self.effective_height - self.agent_h:
                # Landed on the floor (at agent foot level)
                self.powerup_ground_y = self.effective_height - self.agent_h
                self.powerup_falling = False
                self.powerup_on_ground = True

        # --- Update balls (vectorized) ---
        n = self.n_balls
        if n > 0:
            speed_factor = HOURGLASS_SPEED_FACTOR if self.hourglass_active else 1.0
            dt = self.dt

            # Horizontal movement (static balls don't move horizontally)
            moving_mask = self.ball_flags[:n] != BALL_FLAG_STATIC
            self.ball_x[:n][moving_mask] += self.ball_xspeed[:n][moving_mask] * dt * speed_factor

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
            # Uses effective_height for short map levels
            # Bounciness scales the bounce velocity (sqrt so height scales linearly)
            floor_y = self.effective_height
            bottom_edge = self.ball_y[:n] + 2 * self.ball_radius[:n]
            floor_mask = bottom_edge > floor_y
            bounce_speed = self.ball_max_yspeed[:n][floor_mask] * np.sqrt(self.ball_bounciness[:n][floor_mask])
            self.ball_yspeed[:n][floor_mask] = -bounce_speed
            self.ball_y[:n][floor_mask] = floor_y - 2 * self.ball_radius[:n][floor_mask]
            self.ball_chain_depth[:n][floor_mask] = 0  # reset chain on floor contact

            # Update vertical speed (gravity)
            self.ball_yspeed[:n] += self.ball_yacc[:n] * dt * speed_factor
            # Safety clamp — allows chain pop speeds up to 4× normal bounce speed
            max_speed = self.ball_max_yspeed[:n] * 4.0
            self.ball_yspeed[:n] = np.clip(self.ball_yspeed[:n], -max_speed, max_speed)

            # Ceiling check — balls that hit ceiling are destroyed (enables ceiling pops)
            ceiling_mask = self.ball_y[:n] < 0
            if np.any(ceiling_mask):
                indices = np.where(ceiling_mask)[0]
                for idx in sorted(indices, reverse=True):
                    self._remove_ball(idx)
                n = self.n_balls

            # --- Ball-obstacle collisions ---
            if self.n_obstacles > 0 and n > 0:
                self._resolve_ball_obstacles()
                n = self.n_balls

        # --- Check laser-ball collisions ---
        hit_reward, hit_info = self._check_laser_ball_collisions()
        reward += hit_reward
        info["balls_hit"] += hit_info["balls_hit"]
        info["balls_popped"] += hit_info["balls_popped"]

        # --- Check lowering ceiling crushing agent ---
        if self._check_lowering_ceiling_crush():
            reward += REWARDS_GAME_OVER
            terminated = True
            self.done = True
            return reward, terminated, truncated, self._finalize_info(info)

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
                # Reset agent to center, then adjust for door walls
                self.agent_x = (self.width - self.agent_w) / 2.0
                self._position_agent_for_doors()
                # Reset lasers and power-ups for new level
                self.laser_active[:] = False
                self.laser_length[:] = 0.0
                self.laser_max_length[:] = self.effective_height
                self.powerup_on_ground = False
                self.powerup_falling = False
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

    def _position_agent_for_doors(self):
        """Place the agent on the trigger side of any door wall.

        If a door triggers on DOOR_TRIGGER_LEFT, the player must start on
        the left to pop those balls first.  Mirror for TRIGGER_RIGHT.
        """
        for oi in range(self.n_obstacles):
            if self.obs_type[oi] != OBSTACLE_DOOR:
                continue
            trigger = int(self.obs_timer[oi])
            ox = self.obs_x[oi]
            ow = self.obs_w[oi]
            if trigger == DOOR_TRIGGER_LEFT:
                # Place agent on the left side, centred in the left half
                self.agent_x = (ox - self.agent_w) / 2.0
            else:
                # Place agent on the right side
                self.agent_x = ox + ow + (self.width - ox - ow - self.agent_w) / 2.0
            self.agent_x = max(0.0, min(self.agent_x, self.width - self.agent_w))
            break  # only handle the first door

    def _update_dynamic_obstacles(self):
        """Update door walls, opening walls, and lowering ceiling each frame."""
        dt = self.dt
        n = self.n_balls
        i = 0
        while i < self.n_obstacles:
            otype = self.obs_type[i]

            if otype == OBSTACLE_DOOR:
                # Door wall: opens when all balls on the trigger side are cleared.
                # Open door stays as obstacle (blocks balls) but lets agent through.
                # obs_timer >= 0 means closed (value = trigger side), < 0 means open.
                if self.obs_timer[i] >= 0:
                    trigger_side = int(self.obs_timer[i])  # 0=left, 1=right
                    wall_center_x = self.obs_x[i] + self.obs_w[i] / 2.0
                    should_open = True
                    if n > 0:
                        ball_cx = self.ball_x[:n] + self.ball_radius[:n]
                        if trigger_side == DOOR_TRIGGER_LEFT:
                            if np.any(ball_cx < wall_center_x):
                                should_open = False
                        else:  # DOOR_TRIGGER_RIGHT
                            if np.any(ball_cx > wall_center_x):
                                should_open = False
                    if should_open:
                        self.obs_timer[i] = -1.0  # mark as open
                i += 1

            elif otype == OBSTACLE_OPENING:
                # Opening walls: count down timer, then slide apart and remove
                self.obs_timer[i] -= dt
                if self.obs_timer[i] <= 0:
                    # Wall is opening — shrink height from both ends
                    shrink = OPENING_WALL_SPEED * self.height * dt
                    self.obs_y[i] += shrink / 2
                    self.obs_h[i] -= shrink
                    if self.obs_h[i] <= 0:
                        # Fully open — remove this obstacle
                        self._remove_obstacle(i)
                        continue
                i += 1

            elif otype == OBSTACLE_LOWERING_CEIL:
                # Lowering ceiling: descend until reaching target
                target_y = LOWERING_CEIL_TARGET_RATIO * self.height
                if self.obs_h[i] < target_y:
                    growth = LOWERING_CEIL_SPEED_RATIO * self.height * dt
                    self.obs_h[i] = min(self.obs_h[i] + growth, target_y)
                i += 1
            else:
                i += 1

    def _remove_obstacle(self, idx):
        """Remove obstacle at index by swapping with last active obstacle."""
        last = self.n_obstacles - 1
        if idx != last:
            self.obs_x[idx] = self.obs_x[last]
            self.obs_y[idx] = self.obs_y[last]
            self.obs_w[idx] = self.obs_w[last]
            self.obs_h[idx] = self.obs_h[last]
            self.obs_type[idx] = self.obs_type[last]
            self.obs_timer[idx] = self.obs_timer[last]
        self.n_obstacles -= 1

    def _try_fire_laser(self):
        """Fire a laser if a slot is available."""
        agent_center_x = self.agent_x + self.agent_w / 2.0
        for i in range(self.max_lasers):
            if not self.laser_active[i]:
                self.laser_active[i] = True
                self.laser_x[i] = agent_center_x
                self.laser_length[i] = self.agent_h  # Start from agent top
                self.laser_hit_ball[i] = False
                # Compute max reachable length (check obstacles above fire point)
                max_len = float(self.effective_height)
                for oi in range(self.n_obstacles):
                    ox = self.obs_x[oi]
                    ow = self.obs_w[oi]
                    if ox <= agent_center_x <= ox + ow:
                        obs_bottom = self.obs_y[oi] + self.obs_h[oi]
                        reachable = self.effective_height - obs_bottom
                        if reachable > 0:
                            max_len = min(max_len, reachable)
                self.laser_max_length[i] = max_len
                self.shots_fired += 1
                return

    def _update_lasers(self):
        """Grow active lasers upward. Returns wasted shot penalty."""
        penalty = 0.0
        for i in range(self.max_lasers):
            if self.laser_active[i]:
                max_len = self.laser_max_length[i]
                self.laser_length[i] = min(self.laser_length[i] + self.laser_speed, max_len)
                if self.laser_length[i] >= max_len:
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
        danger_y_threshold = 0.7 * self.effective_height

        for laser_idx in range(self.max_lasers):
            if not self.laser_active[laser_idx]:
                continue

            lx = self.laser_x[laser_idx]
            laser_top = self.effective_height - self.laser_length[laser_idx]

            # Ball centers and radii
            bcx = self.ball_x[:n] + self.ball_radius[:n]  # center x
            bcy = self.ball_y[:n] + self.ball_radius[:n]   # center y
            br = self.ball_radius[:n]

            # Check overlap: ball x-range includes laser x, ball y-range overlaps laser y-range
            x_overlap = (bcx - br <= lx) & (bcx + br >= lx)
            y_overlap = (bcy + br >= laser_top) & (bcy - br <= self.effective_height)
            hits = x_overlap & y_overlap

            if not np.any(hits):
                continue

            # Process first hit only (laser deactivates on hit)
            hit_idx = np.where(hits)[0][0]
            lvl = self.ball_level[hit_idx]
            bx = self.ball_x[hit_idx]
            by = self.ball_y[hit_idx]
            b_radius = float(self.ball_radius[hit_idx])
            parent_yspeed        = float(self.ball_yspeed[hit_idx])       # capture before swap-with-last
            parent_bounciness    = float(self.ball_bounciness[hit_idx])
            parent_color         = tuple(self.ball_color[hit_idx])
            parent_chain_depth   = int(self.ball_chain_depth[hit_idx])
            parent_inherit_bounce = bool(self.ball_inherit_bounciness[hit_idx])

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

                # Pop physics: chain-depth-aware impulse + velocity inheritance.
                # Impulse grows with each consecutive split (chain_depth), lighter children
                # (lower level) receive more upward velocity.
                # Formula: impulse = BASE * (1+GAIN)^child_depth / child_level^MASS_EXP
                child_level = lvl - 1
                child_chain_depth = parent_chain_depth + 1
                chain_factor = (1.0 + self.pop_chain_gain) ** child_chain_depth
                impulse = -(self.pop_base_impulse * chain_factor
                            / (child_level ** self.pop_mass_exponent))
                child_yspeed = self.pop_velocity_inherit * parent_yspeed + impulse
                child_bounciness = parent_bounciness if parent_inherit_bounce else 1.0
                self._add_ball(child_level, bx, by, go_right=False,
                               yspeed=child_yspeed, bounciness=child_bounciness,
                               color=parent_color, chain_depth=child_chain_depth)
                self._add_ball(child_level, bx, by, go_right=True,
                               yspeed=child_yspeed, bounciness=child_bounciness,
                               color=parent_color, chain_depth=child_chain_depth)

            # Update n since we modified ball arrays
            n = self.n_balls
            break  # One hit per laser per frame

        return reward, hit_info

    def _pop_ball_at(self, idx):
        """Remove ball at idx (smallest ball, fully destroyed). Returns reward."""
        bx = float(self.ball_x[idx])
        by = float(self.ball_y[idx])
        br = float(self.ball_radius[idx])
        self.recent_pops.append((bx + br, by + br, br))
        self._maybe_drop_powerup(bx, by)
        self._remove_ball(idx)
        return REWARDS_POP_BALL

    def _hit_ball_at(self, idx, level):
        """Remove ball at idx (will split). Returns level-scaled reward."""
        bx = float(self.ball_x[idx])
        by = float(self.ball_y[idx])
        br = float(self.ball_radius[idx])
        self.recent_pops.append((bx + br, by + br, br))
        self._maybe_drop_powerup(bx, by)
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
            self.ball_bounciness[idx] = self.ball_bounciness[last]
            self.ball_flags[idx] = self.ball_flags[last]
            self.ball_color[idx] = self.ball_color[last]
            self.ball_chain_depth[idx] = self.ball_chain_depth[last]
            self.ball_inherit_bounciness[idx] = self.ball_inherit_bounciness[last]
        self.n_balls -= 1

    def _maybe_drop_powerup(self, x, y):
        """Possibly drop a power-up at the given position."""
        if not self.enable_powerups or self.powerup_on_ground or self.powerup_falling:
            return
        if self.rng.random() < POWERUP_DROP_CHANCE:
            self.powerup_falling = True
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
        agent_top = self.effective_height - self.agent_h
        agent_bottom = self.effective_height

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

    def _check_lowering_ceiling_crush(self):
        """Check if the lowering ceiling has reached the agent. Returns True if crushed."""
        agent_top = self.effective_height - self.agent_h
        for oi in range(self.n_obstacles):
            if self.obs_type[oi] == OBSTACLE_LOWERING_CEIL:
                ceil_bottom = self.obs_y[oi] + self.obs_h[oi]
                if ceil_bottom >= agent_top:
                    return True
        return False

    def _check_agent_ball_collision(self):
        """Check if any ball overlaps with the agent's sprite-conforming hitbox.

        The hitbox is 5 stacked rectangles tracing the sprite's hourglass
        silhouette (head, neck, waist, upper legs, lower legs).
        A small forgiveness shrink requires slight overlap before triggering.
        """
        n = self.n_balls
        if n == 0:
            return False

        atop_x = self.agent_x
        atop_y = self.effective_height - self.agent_h

        bcx = self.ball_x[:n] + self.ball_radius[:n]
        bcy = self.ball_y[:n] + self.ball_radius[:n]
        effective_br_sq = np.maximum(0.0, self.ball_radius[:n] - self._hitbox_shrink) ** 2

        hit = np.zeros(n, dtype=bool)
        for (x_off, y_off, w, h) in self._hitbox_rects:
            left = atop_x + x_off
            right = left + w
            top = atop_y + y_off
            bottom = top + h
            closest_x = np.clip(bcx, left, right)
            closest_y = np.clip(bcy, top, bottom)
            dx = bcx - closest_x
            dy = bcy - closest_y
            hit |= (dx * dx + dy * dy) < effective_br_sq

        return np.any(hit)

    def _resolve_ball_obstacles(self):
        """Resolve collisions between active balls and obstacles (minimum-penetration push)."""
        n = self.n_balls
        if n == 0:
            return
        for oi in range(self.n_obstacles):
            ox = self.obs_x[oi]
            oy = self.obs_y[oi]
            ow = self.obs_w[oi]
            oh = self.obs_h[oi]
            ox_right = ox + ow
            oy_bottom = oy + oh

            # Ball bounding boxes (top-left based, diameter = 2*radius)
            bx_left = self.ball_x[:n]
            by_top = self.ball_y[:n]
            bx_right = bx_left + 2 * self.ball_radius[:n]
            by_bottom = by_top + 2 * self.ball_radius[:n]

            # Overlap check
            overlap = ((bx_right > ox) & (bx_left < ox_right)
                       & (by_bottom > oy) & (by_top < oy_bottom))
            if not np.any(overlap):
                continue

            indices = np.where(overlap)[0]
            for idx in indices:
                bl = self.ball_x[idx]
                bt = self.ball_y[idx]
                r2 = 2 * self.ball_radius[idx]
                br = bl + r2
                bb = bt + r2

                # Compute penetration on each axis
                pen_left = br - ox       # ball penetrating from left of obstacle
                pen_right = ox_right - bl  # ball penetrating from right
                pen_top = bb - oy        # ball penetrating from top
                pen_bottom = oy_bottom - bt  # ball penetrating from bottom

                min_pen = min(pen_left, pen_right, pen_top, pen_bottom)

                if min_pen == pen_left:
                    self.ball_x[idx] = ox - r2
                    self.ball_xspeed[idx] = -abs(self.ball_xspeed[idx])
                elif min_pen == pen_right:
                    self.ball_x[idx] = ox_right
                    self.ball_xspeed[idx] = abs(self.ball_xspeed[idx])
                elif min_pen == pen_top:
                    self.ball_y[idx] = oy - r2
                    self.ball_yspeed[idx] = -abs(self.ball_yspeed[idx])
                else:  # pen_bottom
                    self.ball_y[idx] = oy_bottom
                    self.ball_yspeed[idx] = abs(self.ball_yspeed[idx])

    def _resolve_agent_obstacles(self):
        """Push agent out of obstacles that reach near the floor (wall-type).

        Door walls are solid — they block both balls and agent until opened.
        Lowering ceilings don't block agent horizontally.
        """
        if self.n_obstacles == 0:
            return
        agent_left = self.agent_x
        agent_right = self.agent_x + self.agent_w
        agent_top = self.effective_height - self.agent_h

        for oi in range(self.n_obstacles):
            ox = self.obs_x[oi]
            oy = self.obs_y[oi]
            ow = self.obs_w[oi]
            oh = self.obs_h[oi]
            obs_bottom = oy + oh

            # Lowering ceiling doesn't block agent horizontally
            if self.obs_type[oi] == OBSTACLE_LOWERING_CEIL:
                continue

            # Open door (timer < 0) lets agent walk through
            if self.obs_type[oi] == OBSTACLE_DOOR and self.obs_timer[oi] < 0:
                continue

            # Only block agent if obstacle bottom reaches agent's vertical range
            if obs_bottom < agent_top:
                continue

            # Check horizontal overlap
            if agent_right <= ox or agent_left >= ox + ow:
                continue

            # Push agent out along shortest horizontal penetration
            pen_left = agent_right - ox
            pen_right = (ox + ow) - agent_left
            if pen_left < pen_right:
                self.agent_x = ox - self.agent_w
            else:
                self.agent_x = ox + ow

            # Clamp to screen bounds
            self.agent_x = max(0.0, min(self.agent_x, self.width - self.agent_w))

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
            "ball_color": self.ball_color[:self.n_balls].copy(),
            "ball_flags": self.ball_flags[:self.n_balls].copy(),
            "n_balls": self.n_balls,
            "current_level": self.current_level,
            "steps": self.steps,
            "max_steps": self.max_steps,
            "width": self.width,
            "height": self.height,
            "effective_height": self.effective_height,
            "has_double_harpoon": self.has_double_harpoon,
            "has_force_field": self.has_force_field,
            "hourglass_active": self.hourglass_active,
            "hourglass_timer": self.hourglass_timer,
            "powerup_on_ground": self.powerup_on_ground or self.powerup_falling,
            "powerup_ground_x": self.powerup_ground_x,
            "powerup_ground_y": self.powerup_ground_y,
            "powerup_ground_type": self.powerup_ground_type,
            # Obstacles
            "obstacle_x": self.obs_x[:self.n_obstacles].copy(),
            "obstacle_y": self.obs_y[:self.n_obstacles].copy(),
            "obstacle_w": self.obs_w[:self.n_obstacles].copy(),
            "obstacle_h": self.obs_h[:self.n_obstacles].copy(),
            "obstacle_type": self.obs_type[:self.n_obstacles].copy(),
            "obstacle_timer": self.obs_timer[:self.n_obstacles].copy(),
            "n_obstacles": self.n_obstacles,
            # Pop effects (renderer-only)
            "recent_pops": list(self.recent_pops),
        }
