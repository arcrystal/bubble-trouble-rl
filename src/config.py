"""Centralized configuration for Bubble Trouble RL."""

import numpy as np

# ---------------------------------------------------------------------------
# Named color constants (RGB tuples) — use these in LEVEL_DEFS "color" keys
# and import into renderer.py instead of redefining them there.
# ---------------------------------------------------------------------------
BLACK     = (0,   0,   0)
WHITE     = (255, 255, 255)
RED       = (255, 0,   0)
YELLOW    = (245, 237, 7)
GREEN     = (52,  145, 33)
ORANGE    = (237, 141, 45)
BLUE      = (128, 206, 242)
PURPLE    = (160, 50,  220)
BLUEPURPLE   = (144, 125,  230)
CYAN      = (0,   220, 220)
GRAY      = (120, 120, 120)
DARK_GRAY = (60,  60,  60)
DARK_RED  = (180, 30,  30)
CRIMSON   = (140, 20,  60)

# Default ball color per physics level (renderer and engine share this)
BALL_COLORS_BY_LEVEL = {
    1: BLUE,
    2: YELLOW,
    3: GREEN,
    4: ORANGE,
    5: RED,
    6: DARK_RED,
}

# Display dimensions (used for physics scaling)
DEFAULT_WIDTH = 720
DEFAULT_HEIGHT = round(DEFAULT_WIDTH / 1.87)  # 385
DEFAULT_FPS = 60

# Agent dimensions (proportional to display width)
AGENT_WIDTH_RATIO = 30 / DEFAULT_WIDTH
AGENT_HEIGHT_RATIO = 47 / DEFAULT_WIDTH
AGENT_SPEED_DIVISOR = 5.13

# Agent hitbox — 5 stacked rects tracing the sprite's hourglass silhouette.
# Measured from still.png (28×40 sprite scaled to 30×47 agent bbox).
# Each tuple: (x_offset, y_offset, width, height) as ratios of (agent_w, agent_h, agent_w, agent_h).
# Y-ranges tile seamlessly (each zone ends where the next begins).
AGENT_HITBOX_RECTS = [
    (0.213, 0.100, 0.607, 0.275),  # Head        (spy 4-14, full width)
    (0.250, 0.375, 0.500, 0.176),  # Neck        (spy 15-21, medium)
    (0.321, 0.551, 0.357, 0.149),  # Waist       (spy 22-27, narrow)
    (0.250, 0.700, 0.500, 0.126),  # Upper legs  (spy 28-32, medium)
    (0.213, 0.826, 0.607, 0.128),  # Lower legs  (spy 33-37, full width)
]
AGENT_HITBOX_SHRINK = 1.4  # forgiveness pixels at DEFAULT_WIDTH

# Laser
LASER_WIDTH_RATIO = 1 / DEFAULT_HEIGHT

# Ball physics constants per level — measured from reference video (video_physics.json).
# Video play area: 539×273 pixels. Our display: 720×385 pixels.
# All bounce heights are similar (~36-44% of play area height) with longer periods for larger balls.
# Each dict: radius_num/radius_den → radius in pixels, bh_num/bh_den → bounce height, bounce_time in seconds.
# radius = round(radius_num / radius_den * display_width)
# bounce_height = round(bh_num / bh_den * display_height)
# Numerators are 10× the video pixel measurement to preserve sub-pixel precision.
# Denominators are 10× the video play area dimension (5390 for width, 2730 for height).
BALL_LEVELS = {
    1: {"radius_num": 40,  "radius_den": 5390, "bh_num": 61,  "bh_den": 385, "bounce_time": 1.10},
    2: {"radius_num": 72,  "radius_den": 5390, "bh_num": 140, "bh_den": 385, "bounce_time": 1.53},
    3: {"radius_num": 140, "radius_den": 5390, "bh_num": 160, "bh_den": 385, "bounce_time": 1.60},
    4: {"radius_num": 175, "radius_den": 5390, "bh_num": 170, "bh_den": 385, "bounce_time": 1.77},
    5: {"radius_num": 235, "radius_den": 5390, "bh_num": 166, "bh_den": 385, "bounce_time": 2.17},
    6: {"radius_num": 350, "radius_den": 5390, "bh_num": 170, "bh_den": 385, "bounce_time": 2.50},
}
MAX_BALL_LEVEL = max(BALL_LEVELS.keys())  # 6

# Ball colors — measured from reference video (Bubble Struggle 2: Rebubbled).
# Level 1 is BLUE (tiny), not yellow as previously assumed.
BALL_COLORS = {1: "blue", 2: "yellow", 3: "green", 4: "orange", 5: "red", 6: "dark_red"}

# Ball horizontal speed
BALL_XSPEED_DIVISOR = 9.4

# Pop physics — chain-aware split velocity model.
# child_yspeed = INHERIT * parent_yspeed - BASE * (1+CHAIN_GAIN)^child_depth / child_level^MASS_EXP
# child_depth = parent.chain_depth + 1 (resets to 0 on floor bounce).
# Validated constraints (height=385, optimal timing = parent popped near crest):
#   lvl4 chain → lvl1 at depth=3: ~364 px/s → ceiling if ball in top 43% (requires skill)
#   lvl5 chain → lvl1 at depth=4: ~444 px/s → ceiling from top 64% (easier) ✓
#   lvl6 chain → lvl2 at depth=4: ~360 px/s → ceiling if ball in top 40% ✓
POP_BASE_IMPULSE_RATIO = 0.52   # × display_height → base upward velocity at depth=1 (px/s)
POP_VELOCITY_INHERIT = 0.37     # fraction of parent's vertical speed inherited
POP_CHAIN_GAIN = 0.22           # impulse multiplier per chain depth: (1+gain)^depth
POP_MASS_EXPONENT = 0.3         # impulse scales as 1/child_level^exp — lighter = faster

# Maximum balls that can exist simultaneously (2× lvl6 → up to 64 lvl1)
MAX_BALLS = 64

# Ball flags (optional 5th element in level def tuples)
BALL_FLAG_NORMAL = 0
BALL_FLAG_STATIC = 1       # No horizontal motion; children become normal on split

# Obstacles — axis-aligned rectangles: (x_ratio, y_ratio, w_ratio, h_ratio[, type])
# Ratios are fractions of display width/height. (0,0) = top-left.
# Optional 5th element: obstacle type (default OBSTACLE_STATIC).
MAX_OBSTACLES = 8

# Obstacle types
OBSTACLE_STATIC = 0    # Standard solid wall/platform
OBSTACLE_DOOR = 1      # Solid wall that disappears when all balls on one side are cleared
OBSTACLE_OPENING = 2   # Wall that splits apart and disappears after a delay
OBSTACLE_LOWERING_CEIL = 3  # Ceiling bar that descends over time
MAX_OBSTACLE_TYPE = 3  # Largest obstacle type value — used for obs normalization

# Door wall trigger side (6th element in obstacle def tuple for OBSTACLE_DOOR)
DOOR_TRIGGER_LEFT = 0   # Opens when no balls remain to the LEFT of the wall
DOOR_TRIGGER_RIGHT = 1  # Opens when no balls remain to the RIGHT of the wall

# Opening wall: splits apart after this many seconds, then slides away
OPENING_WALL_DELAY_S = 5.0    # Seconds before wall starts opening
OPENING_WALL_SPEED = 1.5      # Ratio of height per second for slide-away speed

# Lowering ceiling: descends from top, stops at target, kills balls/agent on contact
LOWERING_CEIL_TARGET_RATIO = 0.55   # Stops at 55% of play area height from top
LOWERING_CEIL_SPEED_RATIO = 0.04    # Ratio of height per second descent speed

# Per-level play area height overrides (for short maps)
# Value = fraction of normal height. Floor moves up, ceiling stays at y=0.
LEVEL_HEIGHT_OVERRIDE = {
    7: 0.70,  # 70% of normal height — cramped level
}

# Per-level agent start x-position override (as fraction of display width, measured at center).
# Defaults to screen center if not specified.
LEVEL_AGENT_START = {
    8: 0.17,   # Start in leftmost third (center between x=0 and first wall at x≈0.34)
}

OBSTACLE_DEFS = {
    5:  [(0.48, 0.0,  0.04, 0.75),                                    # Solid wall (top 75%)
         (0.48, 0.75, 0.04, 0.25, OBSTACLE_DOOR, DOOR_TRIGGER_LEFT)], # Door (bottom 25%) opens when left cleared
    6:  [(0.0, 0.0, 1.0, 0.02, OBSTACLE_LOWERING_CEIL)],       # Lowering ceiling bar
    7:  [(0.0, 0.0, 1.0, 0.39, OBSTACLE_STATIC)],       # short map — stone ceiling bottom at ~150px; play area 150-270px
    8:  [# Wall 1 (x=0.34): position-based trigger — opens when left section is cleared.
         # Tuple: (x, y, w, h, type, trigger_left_ratio, slide_dir[, section_id])
         # section_id (optional 8th element): if >= 0, wall opens when ALL balls with that
         # ancestry (ball_section == section_id) are destroyed, regardless of position.
         (0.34, 0.00, 0.04, 0.50, OBSTACLE_OPENING, 0.00, -1),      # top half → slides up
         (0.34, 0.50, 0.04, 0.50, OBSTACLE_OPENING, 0.00, +1),      # bottom half → slides down
         # Wall 2 (x=0.66): ancestry-based trigger — opens only when ALL descendants of ball
         # index 1 (orange lvl4) are destroyed, wherever they are on screen. section_id=1.
         (0.66, 0.00, 0.04, 0.50, OBSTACLE_OPENING, 0.38, -1, 1),   # top half → slides up
         (0.66, 0.50, 0.04, 0.50, OBSTACLE_OPENING, 0.38, +1, 1),   # bottom half → slides down
    ],
}

# Power-up types
POWERUP_NONE = 0
POWERUP_LASER_GRID = 2   # Laser fires in a grid pattern (perpendicular crosshairs)
POWERUP_HOURGLASS = 3    # Adds 15 seconds to the level timer

POWERUP_DROP_CHANCE = 0.15          # 15% chance per ball pop
LASER_GRID_STICK_SECONDS = 5.0      # how long a stuck grid laser remains at the ceiling
HOURGLASS_ADD_SECONDS = 15.0        # seconds added to level timer on pickup

# Reward function
REWARDS = {
    # Per-step
    "time_penalty": -0.002,

    # Shooting
    "wasted_shot": -0.3,             # laser reaches ceiling without hitting any ball

    # Ball interactions (level-scaled)
    "hit_ball_base": 0.8,            # per ball_level: reward = 0.8 * level
    "pop_ball": 1.5,                 # smallest ball removed by laser
    "height_bonus_factor": 0.5,      # up to 50% extra reward for hitting balls near ceiling

    # Clutter shaping — guides the agent to declutter before splitting more big balls.
    # On levels with many balls (6, 7, 11, 12) this steers toward clearing existing
    # small balls before creating more, preventing exponential ball-count explosion.
    "clutter_pop_bonus": 0.10,       # extra reward per existing lvl-1 ball (beyond 2) when popping
    "clutter_split_penalty": -0.04,  # penalty per existing lower-level ball (beyond 3) when splitting lvl≥3

    # Shot quality bonus — rewards hitting balls that are both high AND moving upward.
    # height_factor already rewards spatial position; this adds the temporal dimension:
    # children inherit parent's upward velocity, so a fast-rising parent creates faster
    # children with more ceiling-pop potential. Multiplied with height_factor at hit time.
    # Formula: height_factor × upward_factor where upward_factor = |yspeed|/max_yspeed
    # (1.0 = ball moving up at max speed, 0.0 = at apex or falling).
    "shot_quality_scale": 0.25,      # max additional reward at ideal (high+fast-rising) hit

    # Danger awareness
    "danger_split_penalty": -0.2,    # splitting a ball directly above agent when close

    # Level progression (scaled by level: reward *= 1 + (level-1) * scale)
    "finish_level": 10.0,
    "finish_level_scale": 0.25,      # level 1: 10.0, level 6: 22.5, level 12: 37.5
    "time_bonus_scale": 8.0,         # bonus = 8.0 * (remaining_time / max_time) on level clear

    # Terminal (death penalty scaled by levels_cleared: penalty *= 1 + levels_cleared * scale)
    "timeout": -3.0,                 # level timer runs out without clearing
    "game_over": -8.0,               # agent hit by ball (base)
    "game_over_scale": 0.3,          # 0 cleared: -8.0, 5 cleared: -20.0, 11 cleared: -34.4
    "pickup_powerup": 0.3,
    "clear_all_levels": 50.0,
}

# Ceiling pop rewards — when a ball hits the ceiling (y < 0) it's destroyed outright,
# eliminating its entire subtree. Reward = full value of manually clearing all descendants.
# This makes ceiling pops equivalent to manual clearing but better via time/risk savings.
def _ceiling_subtree_value(level, hit_base, pop_val):
    if level == 1:
        return pop_val
    return hit_base * level + 2 * _ceiling_subtree_value(level - 1, hit_base, pop_val)

CEILING_POP_VALUES = {
    L: _ceiling_subtree_value(L, REWARDS["hit_ball_base"], REWARDS["pop_ball"])
    for L in range(1, MAX_BALL_LEVEL + 1)
}
# L=1: 1.5, L=2: 4.6, L=3: 11.6, L=4: 26.4, L=5: 56.8, L=6: 118.4

# Time limit
DEFAULT_LEVEL_TIME_SECONDS = 60.0
DEFAULT_MAX_STEPS = int(DEFAULT_LEVEL_TIME_SECONDS * DEFAULT_FPS)

# Observation space
MAX_OBS_BALLS = 64  # Observation slots for balls — matches MAX_BALLS so agent sees all splits
OBS_PER_BALL = 10   # x, y, xspeed, yspeed, radius, level, is_active, relative_x, peak_height, intercept_x
OBS_AGENT = 5       # x, laser_active, laser_length, laser_x, can_fire
OBS_GLOBAL = 5      # num_balls_ratio, steps_remaining_ratio, current_level, best_chain_x, best_chain_quality
OBS_POWERUP = 6     # has_laser_grid, laser_stuck, powerup_visible, powerup_dist, n_rising_ratio, closest_approach_time
OBS_OBSTACLES = MAX_OBSTACLES * 6  # cx, cy, w, h, type, is_passable per obstacle
OBS_SIZE = MAX_OBS_BALLS * OBS_PER_BALL + OBS_AGENT + OBS_GLOBAL + OBS_POWERUP + OBS_OBSTACLES  # 704

# Level definitions — each level is a list of ball dicts.
# Required keys: "lvl" (1-6), "x" (0-1 ratio of width), "y" (0-1 ratio of height)
# Optional keys:
#   "dir":         "L" or "R"  (default "R")  — initial horizontal direction
#   "static":      bool        (default False) — static until popped: no horizontal motion;
#                              split children revert to normal horizontal motion (BALL_FLAG_STATIC)
#   "bounce":      float       (default 1.0)   — bounciness multiplier
#   "keep_bounce": bool        (default False) — if True, children inherit parent's bounciness on split
#   "color":       RGB tuple   (default = BALL_COLORS_BY_LEVEL[lvl]) — visual color, inherited by children
#                              use named constants: BLUE YELLOW GREEN ORANGE RED DARK_RED PURPLE CYAN …
LEVEL_DEFS = {
    # --- Levels 1-12: measured from per-level reference videos ---
    1: [
        {"lvl": 2, "x": 0.25, "y": 0.25, "dir": "R", "color":YELLOW},                            # 1 yellow ball
    ],
    2: [
        {"lvl": 3, "x": 0.25, "y": 0.25, "dir": "R", "color":GREEN},                            # 1 green ball
    ],
    3: [
        {"lvl": 4, "x": 0.25, "y": 0.25, "dir": "R", "color":RED},                            # 1 orange ball
    ],
    4: [
        {"lvl": 3, "x": 0.25, "y": 0.25, "dir": "L", "color":ORANGE},                            # left orange → wall
        {"lvl": 3, "x": 0.75, "y": 0.25, "dir": "R", "color":ORANGE},                            # right orange → wall
    ],
    5: [
        {"lvl": 3, "x": 0.33, "y": 0.25, "dir": "R", "color":YELLOW},                            # yellow left → door
        {"lvl": 4, "x": 0.65, "y": 0.25, "dir": "L", "color":GREEN},                            # green right → door
    ],
    6: [                                                                            # 6 tiny balls near floor
        {"lvl": 1, "x": 0.01, "y": 0.8, "dir": "L", "color":BLUE},                            # leftmost blue → ceiling
        {"lvl": 1, "x": 0.135, "y": 0.95, "dir": "L", "color":PURPLE},
        {"lvl": 1, "x": 0.26, "y": 0.8, "dir": "L", "color":BLUE},
        {"lvl": 1, "x": 0.385, "y": 0.95, "dir": "L", "color":PURPLE},
        {"lvl": 1, "x": 0.615, "y": 0.95, "dir": "R", "color":BLUE},
        {"lvl": 1, "x": 0.74, "y": 0.8, "dir": "R", "color":PURPLE},
        {"lvl": 1, "x": 0.865, "y": 0.95, "dir": "R", "color":BLUE},
        {"lvl": 1, "x": 0.99, "y": 0.8, "dir": "R", "color":PURPLE},
    ],
    7: [                                                                            # 12 tiny balls, 4 groups of 3 (short map)
        {"lvl": 1, "x": 0.08, "y": 0.67, "dir": "R"},
        {"lvl": 1, "x": 0.10, "y": 0.67, "dir": "R"},
        {"lvl": 1, "x": 0.12, "y": 0.67, "dir": "R"},
        {"lvl": 1, "x": 0.17, "y": 0.67, "dir": "R"},
        {"lvl": 1, "x": 0.19, "y": 0.67, "dir": "R"},
        {"lvl": 1, "x": 0.21, "y": 0.67, "dir": "R"},
        {"lvl": 1, "x": 0.79, "y": 0.67, "dir": "L"},
        {"lvl": 1, "x": 0.81, "y": 0.67, "dir": "L"},
        {"lvl": 1, "x": 0.83, "y": 0.67, "dir": "L"},
        {"lvl": 1, "x": 0.88, "y": 0.67, "dir": "L"},
        {"lvl": 1, "x": 0.90, "y": 0.67, "dir": "L"},
        {"lvl": 1, "x": 0.92, "y": 0.67, "dir": "L"},
    ],
    8: [                                                                            # yellow / orange / red in 3 sections
        {"lvl": 3, "x": 0.18, "y": 0.52, "dir": "R", "color":YELLOW},
        {"lvl": 4, "x": 0.51, "y": 0.30, "dir": "R", "color":ORANGE},
        {"lvl": 5, "x": 0.88, "y": 0.35, "dir": "L", "color":RED},
    ],
    9: [                                                                            # yellow-left, red-center, yellow-right
        {"lvl": 4, "x": 0.13, "y": 0.52, "dir": "L", "color":YELLOW},
        {"lvl": 5, "x": 0.50, "y": 0.31, "static": True, "color":RED},                            # static red ball in center
        {"lvl": 4, "x": 0.85, "y": 0.49, "dir": "R", "color":YELLOW},
    ],
    10: [
        {"lvl": 6, "x": 0.23, "y": 0.30, "dir": "R"},                            # single huge dark-red ball
    ],
    11: [                                                                           # 2 groups of 4 at top
        # static=True → bounce vertically in place until popped; children get normal horizontal motion
        # bounce=2.09 → reach 76% of screen height; keep_bounce=True so children inherit bounciness
        {"lvl": 2, "x": 0.13, "y": 0.24, "static": True, "bounce": 2.09, "keep_bounce": True},
        {"lvl": 2, "x": 0.18, "y": 0.24, "static": True, "bounce": 2.09, "keep_bounce": True},
        {"lvl": 2, "x": 0.23, "y": 0.24, "static": True, "bounce": 2.09, "keep_bounce": True},
        {"lvl": 2, "x": 0.28, "y": 0.24, "static": True, "bounce": 2.09, "keep_bounce": True},
        {"lvl": 2, "x": 0.72, "y": 0.24, "static": True, "bounce": 2.09, "keep_bounce": True},
        {"lvl": 2, "x": 0.77, "y": 0.24, "static": True, "bounce": 2.09, "keep_bounce": True},
        {"lvl": 2, "x": 0.82, "y": 0.24, "static": True, "bounce": 2.09, "keep_bounce": True},
        {"lvl": 2, "x": 0.87, "y": 0.24, "static": True, "bounce": 2.09, "keep_bounce": True},
    ],
    12: [                                                                           # 3 green left + 3 blue right
        {"lvl": 3, "x": 0.1, "y": 0.45, "static": True, "color":PURPLE},
        {"lvl": 3, "x": 0.2, "y": 0.65, "static": True, "color":PURPLE},
        {"lvl": 3, "x": 0.3, "y": 0.70, "static": True, "color":PURPLE},
        {"lvl": 3, "x": 0.7, "y": 0.75, "static": True, "color":BLUEPURPLE},
        {"lvl": 3, "x": 0.8, "y": 0.75, "static": True, "color":BLUEPURPLE},
        {"lvl": 3, "x": 0.9, "y": 0.75, "static": True, "color":BLUEPURPLE},
        {"lvl": 3, "x": 0.25, "y": 0.25, "dir": "R", "color":GREEN},
    ],
}
NUM_LEVELS = len(LEVEL_DEFS)  # 12

# Per-level background colors (RGB tuples) for renderer — sampled from reference video
LEVEL_BACKGROUNDS = {
    1:  (15, 111, 156),     # Blue (sampled from video)
    2:  (161, 23, 3),       # Red/pink gradient (sampled)
    3:  (147, 95, 89),      # Rainbow/pink-ish (sampled)
    4:  (129, 99, 0),       # Dark brown/olive gradient (sampled)
    5:  (97, 71, 103),      # Pink-purple split bg (sampled)
    6:  (121, 82, 43),      # Warm multicolor (sampled)
    7:  (90, 51, 109),      # Purple (sampled)
    8:  (20, 40, 90),       # Deep navy — contrasts with yellow, orange, red split children
    9:  (53, 57, 40),       # Dark green/brown (sampled)
    10: (200, 160, 115),    # Orange/tan (sampled)
    11: (67, 69, 125),      # Blue-purple gradient (sampled)
    12: (127, 86, 70),      # Rainbow (median approximation)
}

# Training hyperparameters
# Benchmarked configs on M1 Max (steps/sec, excluding eval overhead):
#   8env  b256  e10 =>  5,100   (original)
#   16env b2048 e4  => 14,800
#   32env b2048 e4  => 18,000
#   64env b4096 e4  => 23,600
# 16 envs is optimal for 10 CPU cores (8 perf + 2 eff) — avoids context switching.
TRAINING = {
    # Single-phase PPO with extended cycling curriculum.
    # No Phase 2 — the full-game curriculum phase IS the mastery phase.
    "n_envs": 16,
    "total_timesteps": 425_000_000,
    "learning_rate_start": 5e-4,   # Linear decay over full run: 3e-4 → 5e-6
    "learning_rate_end": 5e-6,
    "n_steps": 8192,           # 16 envs × 8192 = 131K buffer — fits full late-level episodes in one GAE pass
    "batch_size": 8192,        # 16 minibatches per epoch
    "n_epochs": 4,             # target_kl already early-stops around epoch 3-4
    "gamma": 0.999,            # ~1000-step effective horizon for ceiling pop chain credit
    "gae_lambda": 0.95,
    "clip_range": 0.15,
    "ent_coef_start": 0.02,    # Decays linearly over full run
    "ent_coef_end": 0.001,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": 0.015,        # Early-stop gradient updates if policy changes too fast
    "net_arch_pi": [1024, 512, 256],  # Capacity for 12 diverse level strategies
    "net_arch_vf": [1024, 512, 256],
}

# RecurrentPPO (LSTM) training hyperparameters — no action masking available,
# so the LSTM must learn can_fire from observation context.
# Longer n_steps critical: LSTM needs long unbroken sequences for hidden state.
RECURRENT_TRAINING = {
    "n_envs": 8,
    "total_timesteps": 120_000_000,
    "learning_rate_start": 3e-4,
    "learning_rate_end": 5e-6,
    "n_steps": 2048,           # LSTM sequence length — shorter than PPO since recurrent buffer is heavier
    "batch_size": 2048,        # Must equal n_steps for RecurrentPPO (full sequences per minibatch)
    "n_epochs": 4,
    "gamma": 0.999,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef_start": 0.02,
    "ent_coef_end": 0.003,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": 0.015,
    "net_arch_pi": [256, 128],    # Smaller MLP before LSTM — LSTM adds its own capacity
    "net_arch_vf": [256, 128],
    "lstm_hidden_size": 128,      # LSTM hidden state size
    "n_lstm_layers": 1,           # Single LSTM layer (more layers rarely help in RL)
}

# Recurrent curriculum: compressed early phases (LSTM needs less data for basics). 120M total.
RECURRENT_CURRICULUM = [
    # --- Progressive expansion (28M steps) ---
    (0,             1, 3,  False),   # 1M  — Learn to shoot and dodge
    (1_000_000,    1, 5,  False),   # 5M  — Door wall obstacle (L5)
    (6_000_000,    3, 8,  False),   # 6M  — Opening walls, static balls
    (12_000_000,    5, 10, False),   # 8M  — Level-6 ball, ceiling pop chains
    (20_000_000,    7, 12, False),   # 8M  — All hard levels, forced exposure
    # --- Full game + hard refresher cycling (92M steps) ---
    (28_000_000,   1, 12, True),    # 8M  — Full sequential game (build end-to-end play)
    (36_000_000,   5, 12, True),    # 8.5M — Hard refresher (maintain L5-12 skills)
    (44_500_000,   1, 12, True),    # 7.5M — Full game polish
    (52_000_000,   9, 12, True),    # 8M  — Hard refresher (L9-12 focus)
    (60_000_000,   1, 12, True),    # 8M  — Full game polish
    (68_000_000,   10, 12, True),   # 8M  — Hard refresher (L10-12 focus)
    (76_000_000,   1, 12, True),    # 8M  — Full game polish
    (84_000_000,   11, 12, True),   # 8M  — Hard refresher (L11-12 focus)
    (92_000_000,   1, 12, True),    # 8M  — Full game polish
    (100_000_000,   12, 12, True),  # 8M  — Hard refresher (L12 only)
    (108_000_000,   1, 12, True),   # 12M — Final full game polish
]

# Cycling curriculum: progressive expansion → full game → hard refresher → full game polish.
# All envs always play the SAME level range (coherent batches, no distribution shock).
# Hard refresher phases (start_level=5+) force practice on levels the agent rarely
# reaches organically, without the value-function corruption of individual-level pinning.
# 425M total.
CURRICULUM = [
    # --- Progressive expansion (100M steps) ---
    (0,             1, 3,  False),   # 5M  — Learn to shoot and dodge
    (5_000_000,    1, 5,  False),   # 10M — Door wall obstacle (L5)
    (15_000_000,    3, 8,  False),   # 25M — Opening walls, static balls
    (40_000_000,    5, 10, False),   # 35M — Level-6 ball, ceiling pop chains
    (75_000_000,    7, 12, False),   # 25M — All hard levels, forced exposure
    # --- Full game + hard refresher cycling (325M steps) ---
    (100_000_000,   1, 12, True),    # 30M — Full sequential game (build end-to-end play)
    (130_000_000,   5, 12, True),    # 25M — Hard refresher (maintain L5-12 skills)
    (155_000_000,   1, 12, True),    # 30M — Full game polish
    (185_000_000,   9, 12, True),    # 25M — Hard refresher (L9-12 focus)
    (210_000_000,   1, 12, True),    # 30M — Full game polish
    (240_000_000,   10, 12, True),   # 25M — Hard refresher (L10-12 focus)
    (265_000_000,   1, 12, True),    # 30M — Full game polish
    (295_000_000,   11, 12, True),   # 25M — Hard refresher (L11-12 focus)
    (320_000_000,   1, 12, True),    # 30M — Full game polish
    (350_000_000,   12, 12, True),   # 25M — Hard refresher (L12 only)
    (375_000_000,   1, 12, True),    # 50M — Final full game polish
]



def compute_ball_properties(level, width, height):
    """Compute radius, max_yspeed, yacc for a ball level at given display dimensions."""
    props = BALL_LEVELS[level]
    radius = round(props["radius_num"] / props["radius_den"] * width)
    bounce_height = round(props["bh_num"] / props["bh_den"] * height)
    bounce_time = props["bounce_time"]
    time_to_peak = bounce_time / 2
    yacc = 2 * bounce_height / time_to_peak ** 2
    max_yspeed = yacc * time_to_peak
    return radius, max_yspeed, yacc
