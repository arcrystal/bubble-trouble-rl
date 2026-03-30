"""Infinity mode spawn schedule and configuration.

The schedule is a list of (delay_seconds, ball_level, side) tuples. Each entry
means: wait `delay_seconds` after the previous spawn, then send a ball of
`ball_level` from `side`. Side is "L" (left), "R" (right), or None (random).

Balls fly in at their natural peak height (the apex of their default bounce).
Color and bounciness are randomized per spawn. Bounciness is capped so that
no ball ever bounces above 75% of the screen height.
"""

# Side constants for readability
L = "L"
R = "R"

# ---------------------------------------------------------------------------
# Spawn schedule: (delay_seconds, ball_level, side)
#
# side: L = left, R = right, None = random
# This defines the rhythm of the game. The schedule loops indefinitely.
# Difficulty comes from accumulation — balls pile up if you don't pop them.
# ---------------------------------------------------------------------------

SPAWN_SCHEDULE = [
    # --- Opening: single ball, learn the ropes ---
    (0,  2, None),   #   0.0s: one ball to start

    # --- Warmup: easy pairs, generous spacing ---
    (6,  2, None),   #   6.0s
    (6,  3, None),   #  12.0s: first medium ball
    (8,  2, None),   #  20.0s
    (1,  2, None),   #  21.0s
    (4,  2, None),   #  25.0s
    (5,  3, None),   #  30.0s
    (2,  2, None),   #  32.0s
    (5,  2, None),   #  37.0s

    # --- Mid game: level 4s appear, moderate pace ---
    (6,  4, None),   #  43.0s: first large ball
    (9,  2, None),   #  52.0s
    (2,  2, None),   #  54.0s
    (1,  2, None),   #  55.0s
    (1,  2, None),   #  56.0s
    (6,  3, None),   #  62.0s
    (2,  2, None),   #  64.0s
    (1,  2, None),   #  65.0s
    (5,  3, None),   #  70.0s
    (2,  3, None),   #  72.0s
    (1,  2, None),   #  73.0s
    (1,  2, None),   #  74.0s
    (8,  4, None),   #  82.0s
    (1,  4, None),   #  83.0s

    # --- Pressure: tempo increases, level 5 arrives ---
    (2,  2, None),   #  85.0s
    (2,  2, None),   #  87.0s
    (2,  5, None),   #  89.0s: first level-5 ball
    (12, 3, None),   # 101.0s
    (4,  3, None),   # 105.0s
    (1,  2, None),   # 106.0s
    (1,  2, None),   # 107.0s
    (12, 5, None),   # 119.0s
    (8,  2, None),   # 127.0s
    (1,  2, None),   # 128.0s
    (2,  3, None),   # 130.0s
    (2,  3, None),   # 132.0s
    (3,  2, None),   # 135.0s
    (1,  2, None),   # 136.0s
    (1,  3, None),   # 137.0s
    (1,  4, None),   # 138.0s

    # --- Endgame: level 6 balls ---
    (16, 6, None),   # 154.0s: first level-6 ball
    (8,  2, None),   # 162.0s
    (1,  2, None),   # 163.0s
    (1,  2, None),   # 164.0s
    (1,  2, None),   # 165.0s
    (1,  2, None),   # 166.0s
    (2,  3, None),   # 168.0s
    (4,  4, None),   # 172.0s
    (8,  5, None),   # 180.0s
    (16, 6, None),   # 196.0s
]

# ---------------------------------------------------------------------------
# Randomization parameters
# ---------------------------------------------------------------------------

# Ball colors — random per spawn (not tied to level). All available colors.
BALL_COLORS = [
    (128, 206, 242),  # blue
    (245, 237,   7),  # yellow
    ( 52, 145,  33),  # green
    (237, 141,  45),  # orange
    (255,   0,   0),  # red
    (180,  30,  30),  # dark red
    (160,  50, 220),  # purple
    (  0, 220, 220),  # cyan
]

# Bounciness: probability and range for high-bounce balls
HIGH_BOUNCE_PROB  = 0.04          # 4% chance per spawn
HIGH_BOUNCE_RANGE = (1.6, 2.5)   # uniform sample if high bounce (capped per level)
BOUNCE_INHERIT_PROB = 0.9        # on split: 90% child keeps parent's high bounciness

# Max bounce height as a fraction of screen height (caps bounciness per level)
MAX_BOUNCE_HEIGHT_RATIO = 0.75

# Spawn safety: don't spawn if agent is within this x-fraction of the spawn edge
SPAWN_SAFETY_RATIO = 0.15

# Max balls the engine can hold (ball slots)
MAX_BALLS = 64
