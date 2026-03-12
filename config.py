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

# Door wall trigger side (6th element in obstacle def tuple for OBSTACLE_DOOR)
DOOR_TRIGGER_LEFT = 0   # Opens when no balls remain to the LEFT of the wall
DOOR_TRIGGER_RIGHT = 1  # Opens when no balls remain to the RIGHT of the wall

# Opening wall: splits apart after this many seconds, then slides away
OPENING_WALL_DELAY_S = 5.0    # Seconds before wall starts opening
OPENING_WALL_SPEED = 0.5      # Ratio of height per second for slide-away speed

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
    7:  [(0.0, 0.0, 1.0, 0.02, OBSTACLE_STATIC)],       # short map with solid ceiling (no lowering)
    8:  [# Wall 1 (x=0.34): each half slides OUT from center when left section is cleared.
         # Tuple: (x, y, w, h, type, trigger_left_ratio, slide_dir)
         # trigger_left_ratio: left boundary of section that must be emptied to open.
         # slide_dir: -1 = slide up off screen, +1 = slide down off screen.
         (0.34, 0.00, 0.04, 0.50, OBSTACLE_OPENING, 0.00, -1),   # top half → slides up
         (0.34, 0.50, 0.04, 0.50, OBSTACLE_OPENING, 0.00, +1),   # bottom half → slides down
         # Wall 2 (x=0.66): opens when middle section (right of wall 1) is cleared.
         # trigger_left = right edge of wall 1 = 0.34+0.04 = 0.38
         (0.66, 0.00, 0.04, 0.50, OBSTACLE_OPENING, 0.38, -1),   # top half → slides up
         (0.66, 0.50, 0.04, 0.50, OBSTACLE_OPENING, 0.38, +1),   # bottom half → slides down
    ],
    13: [(i / 6, 0.0, 0.025, 0.85) for i in range(1, 6)],      # 5 columns
    14: [(0.30, 0.0, 0.03, 0.85), (0.67, 0.0, 0.03, 0.85)],   # 2 columns
    15: [(0.45, 0.0, 0.04, 0.82)],                              # Wall divider
    20: [(0.48, 0.0, 0.04, 0.82)],                              # Center wall
}

# Power-up types
POWERUP_NONE = 0
POWERUP_DOUBLE_HARPOON = 1
POWERUP_FORCE_FIELD = 2
POWERUP_HOURGLASS = 3

POWERUP_DROP_CHANCE = 0.15  # 15% chance per ball pop
HOURGLASS_DURATION_SECONDS = 5.0
HOURGLASS_SPEED_FACTOR = 0.5

# Reward function
REWARDS = {
    # Per-step
    "time_penalty": -0.002,

    # Shooting
    "wasted_shot": -0.30,            # laser reaches ceiling without hitting any ball

    # Ball interactions (level-scaled)
    "hit_ball_base": 0.6,            # per ball_level: reward = 0.6 * level
    "pop_ball": 1.0,                 # smallest ball removed

    # Danger awareness
    "danger_split_penalty": -0.4,    # splitting a ball directly above agent when close

    # Level progression
    "finish_level": 5.0,
    "time_bonus_scale": 3.0,         # bonus = 3.0 * (remaining_time / max_time) on level clear

    # Terminal
    "game_over": -5.0,
    "pickup_powerup": 0.3,
    "clear_all_levels": 15.0,
}

# Time limit
DEFAULT_LEVEL_TIME_SECONDS = 60.0
DEFAULT_MAX_STEPS = int(DEFAULT_LEVEL_TIME_SECONDS * DEFAULT_FPS)

# Observation space
MAX_OBS_BALLS = 16  # Observation slots for balls
OBS_PER_BALL = 6    # x, y, xspeed, yspeed, radius, is_active
OBS_AGENT = 5       # x, laser_active, laser_length, laser_x, can_fire
OBS_GLOBAL = 3      # num_balls_ratio, steps_remaining_ratio, current_level
OBS_POWERUP = 6     # has_double, has_shield, hourglass_active, hourglass_time, powerup_on_ground, powerup_dist
OBS_OBSTACLES = MAX_OBSTACLES * 5  # cx, cy, w, h, type per obstacle
OBS_SIZE = MAX_OBS_BALLS * OBS_PER_BALL + OBS_AGENT + OBS_GLOBAL + OBS_POWERUP + OBS_OBSTACLES  # 150

# Level definitions — each level is a list of ball dicts.
# Required keys: "lvl" (1-6), "x" (0-1 ratio of width), "y" (0-1 ratio of height)
# Optional keys:
#   "dir":         "L" or "R"  (default "R")  — initial horizontal direction
#   "static":      bool        (default False) — static until popped: no horizontal motion;
#                              split children revert to normal horizontal motion (BALL_FLAG_STATIC)
#   "bounce":      float       (default 1.0)   — bounciness multiplier
#   "keep_bounce": bool        (default True)  — if False, children revert to bounce=1.0 on split
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
        {"lvl": 2, "x": 0.05, "y": 0.24, "static": True, "bounce": 2.09, "keep_bounce": True},
        {"lvl": 2, "x": 0.09, "y": 0.24, "static": True, "bounce": 2.09, "keep_bounce": True},
        {"lvl": 2, "x": 0.13, "y": 0.24, "static": True, "bounce": 2.09, "keep_bounce": True},
        {"lvl": 2, "x": 0.22, "y": 0.24, "static": True, "bounce": 2.09, "keep_bounce": True},
        {"lvl": 2, "x": 0.69, "y": 0.24, "static": True, "bounce": 2.09, "keep_bounce": True},
        {"lvl": 2, "x": 0.74, "y": 0.24, "static": True, "bounce": 2.09, "keep_bounce": True},
        {"lvl": 2, "x": 0.78, "y": 0.24, "static": True, "bounce": 2.09, "keep_bounce": True},
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
NUM_LEVELS = len(LEVEL_DEFS)  # 22

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
    10: (254, 208, 144),    # Orange/tan (sampled)
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
    "n_envs": 16,              # SubprocVecEnv: 16 workers on 10 cores
    "total_timesteps": 360_000_000,
    "learning_rate": 3e-4,
    "n_steps": 2048,           # Steps per env per rollout (2048 * 16 = 32768 total per update)
    "batch_size": 4096,        # Larger minibatch for longer rollouts
    "n_epochs": 4,             # 4 epochs: fast, stable for PPO
    "gamma": 0.995,            # Higher gamma — reward for clearing future levels matters more
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.005,         # Lower entropy — less random exploration as curriculum guides difficulty
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "net_arch_pi": [512, 256], # Wider first layer — 142 obs + 22 levels + obstacles need more capacity
    "net_arch_vf": [512, 256],
}

# Curriculum phases: (start_timestep, min_level, max_level, powerups_enabled)
# Iterative progression through all 22 levels over 500M steps.
# Each phase adds 2-4 levels so the agent masters each tier before moving on.
# min_level moves up to keep training focused on the frontier, not replaying easy levels.
CURRICULUM = [
    (0,            1, 2,  False),   # Phase 0: Learn to shoot (lvl1-2, small balls)
    (15_000_000,   1, 4,  False),   # Phase 1: Two-ball levels, lvl3 balls
    (35_000_000,   2, 6,  False),   # Phase 2: First obstacles (lvl5), lvl4 balls
    (60_000_000,   3, 8,  False),   # Phase 3: Wall dividers, multi-size levels
    (90_000_000,   4, 9, False),   # Phase 4: Harder mixed configs
    (120_000_000,  5, 10, True),    # Phase 5: Columns + power-ups
    (160_000_000,  6, 11, True),    # Phase 6: Gauntlet, lane-based levels
    (200_000_000,  7, 12, True),    # Phase 7: Chaotic multi-ball levels
    (250_000_000,  1, 12, True),    # Phase 8: Level-5 balls, obstacle arenas
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
