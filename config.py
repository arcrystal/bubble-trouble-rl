"""Centralized configuration for Bubble Trouble RL."""

import numpy as np

# Display dimensions (used for physics scaling)
DEFAULT_WIDTH = 720
DEFAULT_HEIGHT = round(DEFAULT_WIDTH / 1.87)  # 385
DEFAULT_FPS = 60

# Agent dimensions (proportional to display width)
AGENT_WIDTH_RATIO = 30 / 720
AGENT_HEIGHT_RATIO = 47 / 720
AGENT_SPEED_DIVISOR = 5.13

# Laser
LASER_WIDTH_RATIO = 1 / 500.0

# Ball physics constants per level
# Each tuple: (radius_ratio_num, radius_ratio_den, bounce_height_num, bounce_height_den, bounce_time)
# radius = round(radius_ratio_num / radius_ratio_den * display_width)
# bounce_height = round(bounce_height_num / bounce_height_den * display_height)
BALL_LEVELS = {
    1: {"radius_num": 5, "radius_den": 550, "bh_num": 48, "bh_den": 290, "bounce_time": 1.13},
    2: {"radius_num": 9, "radius_den": 550, "bh_num": 102, "bh_den": 290, "bounce_time": 1.61},
    3: {"radius_num": 17, "radius_den": 550, "bh_num": 125, "bh_den": 290, "bounce_time": 1.78},
    4: {"radius_num": 25, "radius_den": 550, "bh_num": 149, "bh_den": 290, "bounce_time": 1.96},
}

# Ball horizontal speed
BALL_XSPEED_DIVISOR = 9.4

# Maximum balls that can exist simultaneously
MAX_BALLS = 32

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
OBS_SIZE = MAX_OBS_BALLS * OBS_PER_BALL + OBS_AGENT + OBS_GLOBAL + OBS_POWERUP  # 110

# Level definitions: list of (ball_level, x_ratio, y_ratio) tuples
# x_ratio and y_ratio are fractions of display width/height
LEVEL_DEFS = {
    1: [(2, 0.25, 0.25)],
    2: [(3, 0.25, 0.25)],
    3: [(4, 0.25, 0.25)],
    4: [(3, 0.25, 0.25), (3, 0.75, 0.25)],
    5: [(3, 0.33, 0.25), (4, 0.66, 0.25)],
    6: [(1, i / 7, 0.9) for i in range(1, 7)],
    7: ([(1, i / 7 - 40/720, 0.9) for i in range(1, 3)]
        + [(1, i / 7 - 20/720, 0.9) for i in range(1, 3)]
        + [(1, i / 7, 0.9) for i in range(1, 3)]
        + [(1, 5/7 + 10/720, 0.9), (1, 5/7 + 30/720, 0.9), (1, 5/7 + 50/720, 0.9)]
        + [(1, 6/7 + 10/720, 0.9), (1, 6/7 + 30/720, 0.9), (1, 6/7 + 50/720, 0.9)]),
}
NUM_LEVELS = len(LEVEL_DEFS)

# Training hyperparameters
# Benchmarked configs on M1 Max (steps/sec, excluding eval overhead):
#   8env  b256  e10 =>  5,100   (original)
#   16env b2048 e4  => 14,800
#   32env b2048 e4  => 18,000
#   64env b4096 e4  => 23,600
# 16 envs is optimal for 10 CPU cores (8 perf + 2 eff) — avoids context switching.
TRAINING = {
    "n_envs": 16,              # SubprocVecEnv: 16 workers on 10 cores
    "total_timesteps": 250_000_000,
    "learning_rate": 3e-4,
    "n_steps": 1024,           # Steps per env per rollout (1024 * 16 = 16384 total per update)
    "batch_size": 2048,        # Large minibatch: fewer gradient steps = less bottleneck
    "n_epochs": 4,             # 4 epochs (not 10): 2.5x faster, still stable for PPO
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "net_arch_pi": [256, 256],
    "net_arch_vf": [256, 256],
}

# Curriculum phases: (start_timestep, min_level, max_level, powerups_enabled)
CURRICULUM = [
    (0,         1, 2, False),
    (2_000_000, 1, 4, False),
    (5_000_000, 1, 7, False),
    (8_000_000, 1, 7, True),
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
