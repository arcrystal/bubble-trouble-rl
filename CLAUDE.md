# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bubble Trouble arcade game clone (based on Bubble Struggle 2: Rebubbled) with reinforcement learning. A player-controlled agent moves left/right along the floor and shoots a vertical laser upward to pop bouncing balls. Balls split on hit (level 6 → two level 5s → ... → level 1 → destroyed). The goal is to pop all balls across 22 sequential levels without getting hit. Levels may contain wall/platform obstacles that affect ball bouncing, laser reach, and agent movement. An RL agent is trained via PPO to master this.

## Branches

- **main** — primary development branch
- **webapp** — browser-playable version via pygbag (pygame-ce compiled to WebAssembly)

## Setup & Commands

```bash
# Python 3.13, venv recommended
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Play as human (arrow keys + space to shoot, R to restart, Q to quit)
python src/play.py              # start from level 1
python src/play.py 15           # start from level 15

# Train — two-phase: warmup (curriculum) → per-level (all levels in parallel)
python src/train.py                                          # full two-phase run (100M warmup + rest per-level)
python src/train.py --warmup-steps 150000000                 # longer warmup
python src/train.py --warmup-only                            # curriculum only
python src/train.py --skip-warmup --resume checkpoints/warmup_model.zip  # per-level only from checkpoint
python src/train.py --timesteps 1000000 --n-envs 4           # quick smoke test

# Evaluate trained agent
python src/evaluate.py play checkpoints/best/best_model.zip         # visual
python src/evaluate.py play checkpoints/best/best_model.zip --no-render  # headless stats
python src/evaluate.py benchmark                                      # raw env speed

# TensorBoard
tensorboard --logdir ./logs

# Quick sanity check (no test suite exists — use env checker)
python -c "import sys; sys.path.insert(0,'src'); from gymnasium.utils.env_checker import check_env; from bubble_env import BubbleTroubleEnv; check_env(BubbleTroubleEnv())"
```

There is no test suite. Validation is done via `gymnasium.utils.env_checker.check_env()`, manual observation vector inspection, and training smoke tests.

## Architecture

Two-layer design: **headless numpy engine** + **gymnasium wrapper**. Rendering is fully decoupled — training never imports pygame.

All source files live in `src/`. Run scripts as `python src/X.py` from the project root — Python automatically adds `src/` to `sys.path`.

### `src/config.py` — Single source of truth

All constants live here: display dimensions, ball physics tables, reward values, observation layout, level definitions, obstacle definitions, training hyperparameters, curriculum phases, background colors. Nothing is hardcoded elsewhere.

Key sections:
- `BALL_LEVELS` dict — radius/bounce_height/bounce_time per ball level (1-6)
- `MAX_BALL_LEVEL` — derived constant `max(BALL_LEVELS.keys())`, used by engine instead of hardcoded numbers
- `BALL_COLORS` — `{1: "blue", 2: "yellow", 3: "green", 4: "orange", 5: "red", 6: "dark_red"}` (measured from reference video)
- `REWARDS` dict — 10 reward components (see Reward Function below)
- `LEVEL_DEFS` — 22 levels defined as lists of `(ball_level, x_ratio, y_ratio[, bounciness[, flags]])` tuples. Levels 1-12 measured from reference video.
- `OBSTACLE_DEFS` — per-level obstacle rectangles as `(x_ratio, y_ratio, w_ratio, h_ratio[, type])` tuples (levels 5, 6, 8, 13, 14, 15, 20)
- `BALL_FLAG_NORMAL = 0`, `BALL_FLAG_STATIC = 1` — ball behavior flags (5th element in level def)
- `OBSTACLE_STATIC = 0`, `OBSTACLE_DOOR = 1`, `OBSTACLE_OPENING = 2`, `OBSTACLE_LOWERING_CEIL = 3` — obstacle types
- `DOOR_TRIGGER_LEFT = 0`, `DOOR_TRIGGER_RIGHT = 1` — which side must be cleared to open a door wall
- `OPENING_WALL_DELAY_S = 5.0`, `OPENING_WALL_SPEED = 0.5` — opening wall timer and slide speed
- `LOWERING_CEIL_TARGET_RATIO = 0.55`, `LOWERING_CEIL_SPEED_RATIO = 0.04` — ceiling descent config
- `LEVEL_HEIGHT_OVERRIDE` — per-level play area height overrides for short maps (level 7: 70%)
- `MAX_OBSTACLES = 8`, `MAX_BALLS = 64`
- `TRAINING` — PPO hyperparameters (n_envs=16, n_steps=2048, batch_size=4096, n_epochs=4, 512x256 MLP, gamma=0.995, 500M total_timesteps)
- `CURRICULUM` — 11 phases: `(start_timestep, min_level, max_level, powerups_enabled)` — iterative level-by-level progression
- `LEVEL_BACKGROUNDS` — per-level RGB background colors for renderer
- `compute_ball_properties(level, width, height)` — derives radius, max_yspeed, yacc from kinematic equations

### `src/engine.py` — `BubbleTroubleEngine`

Pure-numpy physics engine. Zero pygame dependency. All ball state lives in preallocated numpy arrays of shape `(MAX_BALLS=64,)`. Obstacle state in arrays of shape `(MAX_OBSTACLES=8,)`. Updates are vectorized where possible.

**Ball physics**: 6 ball levels (measured from reference video — all bounce to similar heights ~36-44% of play area, with longer periods for larger balls). Kinematic equations. Gravity derived from `bounce_height` and `bounce_time` per ball level: `yacc = 2 * bounce_height / (bounce_time/2)^2`. Balls bounce off walls (horizontal), floor (vertical, reset to `-max_yspeed * sqrt(bounciness)`), and obstacles (minimum-penetration-axis push). Balls that reach the ceiling (`y < 0`) are **destroyed outright** regardless of size — this is the ceiling pop mechanic. Each ball has a `bounciness` property (default 1.0, configurable per ball in level defs) that scales floor-bounce speed. Balls can have `flags`: `BALL_FLAG_STATIC` balls have zero horizontal speed (they bounce vertically in place). Static balls' children inherit `BALL_FLAG_NORMAL` on split. All spatial values scale proportionally to display dimensions.

**Ball splitting & pop physics**: Children receive a chain-depth-aware upward impulse plus inheritance of the parent's vertical velocity. Each ball tracks `ball_chain_depth` — how many consecutive splits happened in its ancestry without touching the floor. The chain resets to 0 on any floor bounce.

Formula: `child_yspeed = POP_VELOCITY_INHERIT * parent_yspeed - BASE * (1+CHAIN_GAIN)^child_depth / child_level^MASS_EXP`

Where `child_depth = parent.chain_depth + 1`. Key constants: `POP_VELOCITY_INHERIT = 0.37`, `POP_BASE_IMPULSE_RATIO = 0.584` (×height), `POP_CHAIN_GAIN = 0.25`, `POP_MASS_EXPONENT = 0.3`. Validated ceiling-pop constraints (parent popped near crest, from ~y=150):
- Level 4 chain → lvl1 at depth=3: ~439 px/s → reaches ceiling ✓
- Level 5 chain → lvl1 at depth=4: ~549 px/s → easier ✓
- Level 6 chain → lvl2 at depth=4 (6→5→4→3→2): ~446 px/s → reaches ceiling ✓

Children inherit parent's bounciness. The velocity clamp allows up to 4× normal bounce speed to support deep-chain momentum.

**Ball removal**: Uses **swap-with-last** for O(1) deletion from numpy arrays. `_remove_ball(idx)` copies the last active ball into slot `idx`, then decrements `n_balls`. **Critical pattern**: any data you need from the ball at `idx` must be captured BEFORE calling `_remove_ball()` or `_hit_ball_at()`, because the array slot will contain different data afterward.

**Obstacles**: Axis-aligned rectangles loaded per level from `OBSTACLE_DEFS`. Four obstacle types:
- `OBSTACLE_STATIC` (0): Standard solid wall/platform
- `OBSTACLE_DOOR` (1): Solid wall that blocks balls and agent when closed. When all balls on the trigger side are cleared, the door opens: it stays visible (dimmer shade) and still blocks balls, but lets the agent walk through. `obs_timer >= 0` = closed (value = trigger side), `obs_timer < 0` = open. Trigger side stored as 6th element in obstacle def (`DOOR_TRIGGER_LEFT`/`DOOR_TRIGGER_RIGHT`). In level 5, the door opens when the left ball is fully popped.
- `OBSTACLE_OPENING` (2): Wall that starts solid, then after `OPENING_WALL_DELAY_S` splits apart and slides away (level 8). Removed from obstacle list when fully open.
- `OBSTACLE_LOWERING_CEIL` (3): Full-width ceiling bar that descends at `LOWERING_CEIL_SPEED_RATIO` per second, stops at `LOWERING_CEIL_TARGET_RATIO` (level 6). Compresses play area, making ball dodging harder. Kills agent if ceiling reaches agent top.

Three collision systems:
- **Ball-obstacle** (`_resolve_ball_obstacles`): minimum-penetration-axis push — balls bounce off obstacle surfaces like walls. Balls are fully blocked by all obstacle types.
- **Laser-obstacle** (in `_try_fire_laser`): precomputes `laser_max_length` — the laser stops at the bottom of any obstacle directly above the fire point instead of reaching the ceiling.
- **Agent-obstacle** (`_resolve_agent_obstacles`): only obstacles whose bottom reaches the agent's vertical range block the agent horizontally. Lowering ceilings don't block agent horizontally. Open doors (timer < 0) let agent pass but still block balls. Lowering ceilings don't block agent horizontally.

**Short map**: Levels with entries in `LEVEL_HEIGHT_OVERRIDE` have a raised floor (`effective_height < height`). Balls bounce off the effective floor, agent stands on it, lasers fire from it. Currently level 7 uses 70% height.

**Laser lifecycle**: Fires from agent center, grows upward at `height/fps` pixels per frame (~60 frames = 1 second to reach ceiling). If it hits a ball, it deactivates immediately. If it reaches its max length (ceiling or obstacle bottom) without hitting anything, it's a wasted shot (penalty). `laser_hit_ball[]` array tracks whether each laser hit anything before deactivating. `laser_max_length[]` tracks per-laser ceiling (accounting for obstacles above).

**Power-ups**: 15% drop chance per ball pop. Three types: double harpoon (2 simultaneous lasers), force field (survive one collision), hourglass (slow balls 50% for 5 seconds). Power-ups fall from the ball's pop position to the floor before becoming pickable. Only one power-up can exist at a time (checked via `powerup_on_ground or powerup_falling`).

**Level progression**: Sequential levels 1-22. When all balls are cleared, advance to next level. Steps reset per level (each level gets a full 60-second timer). Lasers, power-ups, and obstacles are cleared/reloaded on level transition.

**Pop effects tracking**: `recent_pops` list collects `(x, y, radius)` for balls destroyed/split each step. Included in `get_state()` for renderer star-burst effects. Cleared at start of each `step()`.

**`_finalize_info(info)`**: Helper that stamps cumulative episode counters (`shots_wasted`, `danger_splits`, `levels_cleared`, `highest_level`) onto the info dict. Called before every return in `step()` to ensure consistent data regardless of which code path exits.

**`step()` return paths** (4 total):
1. Already done → `(0, True, False, info)`
2. Agent-ball collision (death) → `(reward, True, False, info)`
3. All levels cleared → `(reward, False, True, info)`
4. Normal/timeout → `(reward, False, truncated, info)`

### `src/bubble_env.py` — `BubbleTroubleEnv(gymnasium.Env)`

Wraps the engine as a standard Gymnasium environment.

**Action space**: `MultiDiscrete([3, 2])` — movement (0=LEFT, 1=RIGHT, 2=STILL) × shooting (0=SHOOT, 1=NO_SHOOT). Allows simultaneous move + shoot.

**Observation space**: `Box(low=-1, high=1, shape=(174,), dtype=float32)`. All features normalized to [-1, 1]. With frame stacking (n=4), training sees shape `(696,)`.

Layout (174 elements):

| Index | Count | Feature |
|-------|-------|---------|
| 0-15 | 16 | Ball center x (sorted by horizontal distance to agent) |
| 16-31 | 16 | Ball center y |
| 32-47 | 16 | Ball x-speed |
| 48-63 | 16 | Ball y-speed |
| 64-79 | 16 | Ball radius |
| 80-95 | 16 | Ball level (normalized by MAX_BALL_LEVEL) |
| 96-111 | 16 | Ball is_active flag (1.0 if ball exists, 0.0 if empty slot) |
| 112 | 1 | Agent center x |
| 113 | 1 | Laser active (1.0 / -1.0) |
| 114 | 1 | Longest laser length |
| 115 | 1 | Laser x position (where the laser is horizontally) |
| 116 | 1 | Can-fire (1.0 if SHOOT would fire, -1.0 if all slots busy) |
| 117 | 1 | Ball count ratio |
| 118 | 1 | Time remaining ratio |
| 119 | 1 | Current level (auto-scales with NUM_LEVELS) |
| 120 | 1 | Has laser grid (1.0 / -1.0) |
| 121 | 1 | Laser stuck at ceiling (1.0 / -1.0) |
| 122 | 1 | Power-up visible on ground or falling (1.0 / -1.0) |
| 123 | 1 | Signed distance to power-up |
| 124-125 | 2 | Reserved |
| 126-173 | 48 | Obstacle features: 8 slots x 6 values (center_x, center_y, width, height, type, is_passable) |

**Action masking**: `action_masks()` returns flat bool array of shape `(5,)` for `MultiDiscrete([3, 2])`: `[LEFT, RIGHT, STILL, SHOOT, NO_SHOOT]`. SHOOT is masked when no laser slot is available. Used by `MaskablePPO` from sb3-contrib.

Ball slots are sorted by horizontal distance to agent center. Empty ball/obstacle slots are zero-filled.

**Normalization**: Ball yspeed and radius use **fixed normalizers** (`max_possible_yspeed` = 4× max ball level bounce speed to cover chain pop velocities, `max_possible_radius` from max ball level) rather than dynamic normalization. This ensures stationary observation statistics regardless of which balls are alive.

**`set_curriculum()`**: Called by the training callback to adjust `start_level`, `max_level`, and `enable_powerups` between episodes.

### `src/train.py` — MaskablePPO Training

Uses sb3-contrib `MaskablePPO` with `SubprocVecEnv` for parallel environments. Default: 500M timesteps, two-phase strategy (curriculum warmup → per-level fine-tuning).

**Training stack** (wrapper order matters):
1. `BubbleTroubleEnv` — base gymnasium env
2. `ActionMasker` — wraps env to expose `action_masks()` for MaskablePPO
3. `Monitor` — episode stats tracking
4. `SubprocVecEnv` / `DummyVecEnv` — parallelization
5. `VecNormalize(norm_obs=False, norm_reward=True)` — reward normalization only (obs already in [-1,1])
6. `VecFrameStack(n_stack=4)` — temporal context for ball trajectory estimation

**Schedules**: Learning rate and entropy coefficient both linearly decay over training:
- LR: 3e-4 → 1e-5
- Entropy: 0.01 → 0.001

**Callbacks**:
- `CurriculumCallback` — advances difficulty based on `CURRICULUM` phases in config. Unwraps through VecNormalize/VecFrameStack to reach base vec env.
- `MetricsCallback` — logs game stats to TensorBoard **only at episode boundaries** (when SB3's Monitor wrapper adds the `"episode"` key to info)
- `EvalCallback` — periodic evaluation with deterministic policy, saves best model
- `CheckpointCallback` — periodic model snapshots

**Device**: CPU is faster than MPS/GPU for small MLP policies. Script defaults to CPU unless CUDA is available.

**Saved artifacts**: `final_model.zip` (model weights) + `vecnormalize.pkl` (reward normalization stats, needed for evaluation).

### `src/renderer.py` — `PygameRenderer`

Optional pygame-ce renderer. Only imported lazily by `bubble_env.py` when `render_mode` is set. Reads the engine's `get_state()` dict and draws everything. Not used during training.

Visual features:
- Per-level background colors from `LEVEL_BACKGROUNDS`
- Ball colors: blue(1), yellow(2), green(3), orange(4), red(5), dark red(6)
- Thin gray/white laser lines
- Stone-colored obstacle rectangles with dark borders; closed door walls in distinct shade, open doors in dimmer shade
- Static balls shown with thin white outline ring
- Timer bar at bottom of play area (red depleting bar)
- HUD score bar: "PLAYER 1 ... LEVEL N ... PLAYER 2"
- "Get ready" splash for first 90 frames of each level
- Star-burst pop effects on ball destruction
- Force field cyan border on agent

### `src/play.py` — Human Play

Keyboard-playable game using the engine + renderer. Arrow keys to move, space to shoot, R to restart, Q to quit. Accepts optional level number argument: `python play.py 15`.

### `src/evaluate.py` — Evaluation & Benchmarking

Two subcommands: `play` (load model, run episodes, print stats) and `benchmark` (random actions, measure steps/sec).

## 22 Levels

Levels 1-12 are measured from the reference video (Bubble Struggle 2: Rebubbled). Levels 13-22 are designed for difficulty progression.

| Level | Balls | Obstacles | Notes |
|-------|-------|-----------|-------|
| 1 | 1x lvl2 (yellow) | none | Single small ball |
| 2 | 1x lvl3 (green) | none | Single medium ball |
| 3 | 1x lvl5 (red) | none | Single large ball |
| 4 | 2x lvl4 (orange) | none | Two balls |
| 5 | 1x lvl2 + 1x lvl3 | door wall | Door opens when left ball fully popped |
| 6 | 8x lvl1 (blue) | lowering ceiling | Ceiling descends, compresses play area |
| 7 | 12x lvl1 (4 groups of 3) | short map (70%) + lowering ceiling | Raised floor + descending ceiling |
| 8 | 1x lvl2 + 1x lvl4 + 1x lvl5 | 2 opening walls | Walls disappear after 5s |
| 9 | 1x lvl5 + 2x lvl2 | none | Red center + yellow sides |
| 10 | 1x lvl6 (dark red) | none | Single huge ball |
| 11 | 4x lvl3 + 4x lvl2 | none | Many balls at top |
| 12 | 3x lvl3 + 3x lvl1 | none | Mixed near floor |
| 13 | 6 balls across lanes | 5 columns | Lane-based level |
| 14 | 3 balls | 2 columns | Column level |
| 15 | 1x lvl4 + 1x lvl3 | wall divider | Hard obstacle level |
| 16 | 8x lvl1 in a row | none | Dodging gauntlet |
| 17 | 1x lvl3 + 1x lvl1 + 1x lvl2 | none | Mixed |
| 18 | lvl4+lvl3+smaller | none | Chaotic mix |
| 19 | 2x lvl5 | none | Introduces lvl5 |
| 20 | 1x lvl5 + 1x lvl3 | center wall | Lvl5 in split arena |
| 21 | 1x lvl6 + 1x lvl4 | none | Introduces lvl6 |
| 22 | 1x lvl6 + 1x lvl5 | none | Final boss |

## Reference Videos — `game_levels/`

Individual level completion videos from the real Bubble Struggle 2: Rebubbled game. These are the **ground truth** for configuring `LEVEL_DEFS`, `OBSTACLE_DEFS`, `LEVEL_BACKGROUNDS`, `LEVEL_HEIGHT_OVERRIDE`, ball physics, and obstacle behavior. All videos are 640×360 @ 30fps.

**Play area dimensions** (in video pixels): left=51, right=590, top=20, bottom=293 → 539×273px play area. Scale to our game: 720/539 = 1.336× width, 385/273 = 1.410× height. Ball radii and positions measured in video pixels must be scaled by these factors.

### Per-Level Video Reference

| File | Duration | Level Summary |
|------|----------|---------------|
| `level01.mp4` | 9s | 1x yellow ball (lvl 2), ~15% from left, mid-height. Blue bg. No obstacles. |
| `level02.mp4` | 13s | 1x green ball (lvl 3), ~25% from left. Red/pink gradient bg. No obstacles. |
| `level03.mp4` | 18s | 1x red ball (lvl 5), ~20% from left, near top. Rainbow gradient bg. No obstacles. |
| `level04.mp4` | 19s | 2x orange balls (lvl 4), ~20% left + ~80% right, mid-height. Dark brown/green bg. |
| `level05.mp4` | 38s | 1x yellow (lvl 2) left + 1x green (lvl 3) right. Center door wall (solid top ~75%, door bottom ~25%). Pink/blue split bg. Door opens when left ball fully popped → agent walks through. Balls blocked by door even when open. |
| `level06.mp4` | 15s | 8x tiny blue balls (lvl 1) in a row near floor. Lowering ceiling (purple zigzag) descends from top. Multicolor/warm bg. Short map visible (reduced play area height). |
| `level07.mp4` | 14s | **Short map** with **lowering ceiling** (purple zigzag). 12 tiny balls in 4 groups of 3 along a horizontal line. Ball colors: 6 red, 4 orange, 2 yellow — all tiny/small size. Purple bg. Groups spaced evenly across width. Ceiling compresses play area over time. |
| `level08.mp4` | 65s | 3 sections divided by 2 vertical walls. Left: 1x yellow (lvl 2). Center: 1x orange (lvl 4). Right: 1x red (lvl 5). Walls are **opening type** — they disappear sequentially as sections are cleared. Cyan/pink/rainbow bg. |
| `level09.mp4` | 69s | 1x red ball (lvl 5) center + 2x yellow balls (lvl 2) on left and right sides. Green/brown bg. No obstacles. |
| `level10.mp4` | 76s | 1x very large dark red ball (lvl 6). Orange/tan bg. No obstacles. Long video due to many split generations. |
| `level11.mp4` | 32s | 2 groups of 4 balls at top of screen. Left group: 2 green (lvl 3) + 2 yellow (lvl 2). Right group: same. Total: 4 green + 4 yellow. Blue/purple gradient bg. |
| `level12.mp4` | 71s | 3x green balls (lvl 3) on left side at staggered heights + 3x tiny blue balls (lvl 1) on right side near floor. Rainbow bg. Ball colors can appear shifted against rainbow background. |

### Key Observations from Videos

**Level 5 door wall**: The wall is a full-height stone column with a door opening at the bottom (~25% of wall height). When closed, it blocks both agent and balls. When the left side's balls are all popped, the door opens: agent can walk through, but balls remain blocked. The door stays visible in a dimmer shade.

**Level 6 & 7 lowering ceiling**: A purple zigzag bar descends from the top of the play area. It compresses the available bounce space. In level 7, the short map + lowering ceiling makes the play area very cramped. The ceiling descends steadily over the duration of the level.

**Level 7 ball colors vs levels**: The 12 balls are all tiny/small size despite being red, orange, and yellow. In the real game these may be decorative colors. In our engine, ball color = ball level, so these need to be mapped to levels that produce the correct tiny size. Currently uncertain whether these are all lvl 1 (same size, different decorative colors) or a mix of lvl 1 + lvl 2.

**Level 8 opening walls**: The 2 vertical walls start solid. When all balls in a section are cleared, the adjacent wall opens (slides/disappears). The player must clear sections one at a time, left to right. This creates a sequential section-clearing mechanic.

**Level 11 ball arrangement**: Two symmetric groups at the very top of the screen. Each group has green balls on the left and yellow balls on the right, tightly packed. They start bouncing immediately.

**Level 12 ball composition**: 3 green balls (lvl 3) on the left side at staggered heights (x=0.08/y=0.45, x=0.20/y=0.65, x=0.27/y=0.70) plus 3 tiny blue balls (lvl 1) on the right side near the floor (y=0.75). Against the rainbow background, ball colors can appear shifted (purple/violet tint).

### Using These Videos

When adjusting `LEVEL_DEFS` or `OBSTACLE_DEFS` in `config.py`, extract frames from the relevant video:
```bash
# First frame (usually "Get Ready" screen showing initial layout)
ffmpeg -i game_levels/levelNN.mp4 -vf "select=eq(n\,0)" -vframes 1 -y /tmp/levelNN_frame0.png

# Gameplay frame at T seconds
ffmpeg -i game_levels/levelNN.mp4 -ss T -vframes 1 -y /tmp/levelNN_tT.png

# Extract ball positions from frame using Python + OpenCV
# Convert video pixel coords to game ratios: x_ratio = (x_px - 51) / 539, y_ratio = (y_px - 20) / 273
```

Ball position ratios in `LEVEL_DEFS` are fractions of game width/height. To convert from video measurements:
- `x_ratio = (ball_center_x_px - 51) / 539` (within 539px-wide play area)
- `y_ratio = (ball_center_y_px - 20) / 273` (within 273px-tall play area)

## Reward Function

Designed to make strategic shooting the dominant strategy. All values in `config.py:REWARDS`.

| Component | Value | Trigger |
|-----------|-------|---------|
| `time_penalty` | -0.002 | Every step |
| `wasted_shot` | -0.30 | Laser reaches ceiling/obstacle without hitting any ball |
| `hit_ball_base` | 0.6 * level | Ball split (level 2-6) |
| `pop_ball` | 1.0 | Smallest ball (level 1) destroyed |
| `danger_split_penalty` | -0.4 | Splitting a ball that's close to agent and low on screen |
| `finish_level` | 5.0 | All balls cleared on current level |
| `time_bonus_scale` | 3.0 * (remaining/max) | Bonus for faster clears |
| `game_over` | -5.0 | Agent hit by ball (death) |
| `pickup_powerup` | 0.3 | Agent picks up a ground power-up |
| `clear_all_levels` | 15.0 | Beat all 22 levels |

The break-even accuracy for shooting is ~30% — the agent must hit at least 1 in 3 shots for shooting to be profitable. This eliminates laser spamming.

## Curriculum Learning

11 phases defined in `config.py:CURRICULUM` (500M total timesteps). Each phase adds 2-4 levels so the agent masters each tier before expanding. `min_level` advances to keep training focused on the frontier.

| Phase | Timestep | Levels | Power-ups | Focus |
|-------|----------|--------|-----------|-------|
| 0 | 0 | 1-2 | Off | Learn to shoot |
| 1 | 15M | 1-4 | Off | Two-ball levels, lvl3 balls |
| 2 | 35M | 2-6 | Off | First obstacles (lvl5), lvl4 balls |
| 3 | 60M | 3-8 | Off | Wall dividers, multi-size |
| 4 | 90M | 4-11 | Off | Harder mixed configs |
| 5 | 120M | 5-14 | On | Columns + power-ups |
| 6 | 160M | 7-16 | On | Gauntlet, lane-based |
| 7 | 200M | 9-18 | On | Chaotic multi-ball |
| 8 | 250M | 11-20 | On | Level-5 balls, obstacle arenas |
| 9 | 310M | 14-22 | On | Level-6 balls, endgame |
| 10 | 380M | 1-22 | On | Full mastery |

## Known Patterns & Pitfalls

### Swap-with-last array mutation
`_remove_ball(idx)` overwrites slot `idx` with the last active ball's data, then decrements `n_balls`. Any read of `self.ball_*[idx]` after removal gets the WRONG ball. Always capture needed values before calling `_remove_ball()`, `_hit_ball_at()`, or `_pop_ball_at()`.

### Info dict consistency
`step()` has 4 return paths. The `_finalize_info()` helper ensures all paths report the same cumulative counters. If adding new counters, add them to `_finalize_info()`, not inline.

### SB3 Monitor episode logging
SB3's Monitor wrapper adds an `"episode"` key to info only on the terminal step. Metrics that should be per-episode (not per-step) must check for this key before logging. See `MetricsCallback._on_step()`.

### Fixed vs dynamic normalization
Observation features that normalize by "what's currently on screen" create non-stationary statistics that hurt the value function. Use fixed normalizers: `max_possible_yspeed` = 4× max ball level bounce speed (covers chain pop velocities up to depth ~5), `max_possible_radius` from max ball level properties.

### Obstacle collision ordering
Ball-obstacle collisions run after ball movement + wall bounces + ceiling check, but before laser-ball and agent-ball collision checks. Agent-obstacle collisions run immediately after agent movement, before any other physics.

### Laser max length with obstacles
Each laser computes its max reachable length at fire time based on obstacles directly above the fire point. If an obstacle is overhead, the laser stops at the obstacle's bottom edge. This is stored in `laser_max_length[i]` and used in `_update_lasers()` instead of `self.height`.

### Ball level extensibility
All ball level references use `MAX_BALL_LEVEL` from config, never hardcoded numbers. The `_load_random_level()` weight formula `2^lvl - 1` auto-scales. To add more ball levels, just add entries to `BALL_LEVELS` in config.py.

### Breaking changes (from original 7-level version)
- Observation space: 110 → 150 elements (40 obstacle features: 8 slots × 5 values including type)
- Ball levels: 4 → 6 (fixed normalizers now use level-6 values)
- Ball colors reordered: lvl1=blue, lvl2=yellow, lvl3=green (measured from reference video)
- Pop physics: chain-depth-aware model (BASE=0.584×h, CHAIN_GAIN=0.25, MASS_EXP=0.3); `ball_chain_depth` array added; clamp raised to 4×
- MAX_BALLS: 32 → 64
- NUM_LEVELS: 7 → 22
- Levels 1-12 redefined from reference video measurements
- Ball flags system added (BALL_FLAG_STATIC for non-moving balls)
- Obstacle type system added (OBSTACLE_DOOR, OBSTACLE_OPENING, OBSTACLE_LOWERING_CEIL)
- Short map system added (LEVEL_HEIGHT_OVERRIDE for raised floor levels)
- Dynamic obstacles: opening walls slide apart, lowering ceiling descends over time
- All previously trained models are incompatible
