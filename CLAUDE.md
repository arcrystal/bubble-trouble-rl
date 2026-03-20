# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bubble Trouble arcade game clone (based on Bubble Struggle 2: Rebubbled) with reinforcement learning. A player-controlled agent moves left/right along the floor and shoots a vertical laser upward to pop bouncing balls. Balls split on hit (level 6 → two level 5s → ... → level 1 → destroyed). The goal is to pop all balls across 12 sequential levels without getting hit. Levels may contain wall/platform obstacles that affect ball bouncing, laser reach, and agent movement. An RL agent is trained via PPO to master this.

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

# Train — MaskablePPO (action-masked, feedforward MLP)
python src/train_ppo.py                                          # full 300M-step run
python src/train_ppo.py --timesteps 5000000                      # shorter run
python src/train_ppo.py --timesteps 100000 --n-envs 4            # quick smoke test
python src/train_ppo.py --resume checkpoints/ppo/periodic/ppo_XXX.zip --vecnorm checkpoints/ppo/periodic/ppo_XXX_vecnorm.pkl  # resume

# Train — RecurrentPPO (LSTM policy, no action masking)
python src/train_recurrent.py                                    # full 300M-step run
python src/train_recurrent.py --timesteps 100000 --n-envs 4      # quick smoke test
python src/train_recurrent.py --resume checkpoints/recurrent/periodic/recurrent_ppo_XXX.zip --vecnorm ...

# Evaluate trained agent
python src/evaluate.py play checkpoints/ppo/best/best_model.zip         # visual
python src/evaluate.py play checkpoints/ppo/best/best_model.zip --no-render  # headless stats
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
- `REWARDS` dict — 12 reward components (see Reward Function below)
- `CEILING_POP_VALUES` — recursive subtree reward for ceiling-popped balls (L1:1.5 → L6:118.4)
- `LEVEL_DEFS` — 12 levels defined as dicts with keys `lvl`, `x`, `y`, `dir`, `static`, `bounce`, `keep_bounce`, `color`. Levels 1-12 measured from reference video.
- `OBSTACLE_DEFS` — per-level obstacle rectangles as `(x_ratio, y_ratio, w_ratio, h_ratio[, type])` tuples (levels 5, 6, 8, 13, 14, 15, 20)
- `BALL_FLAG_NORMAL = 0`, `BALL_FLAG_STATIC = 1` — ball behavior flags (5th element in level def)
- `OBSTACLE_STATIC = 0`, `OBSTACLE_DOOR = 1`, `OBSTACLE_OPENING = 2`, `OBSTACLE_LOWERING_CEIL = 3` — obstacle types
- `DOOR_TRIGGER_LEFT = 0`, `DOOR_TRIGGER_RIGHT = 1` — which side must be cleared to open a door wall
- `OPENING_WALL_DELAY_S = 5.0`, `OPENING_WALL_SPEED = 1.5` — opening wall timer and slide speed
- `LOWERING_CEIL_TARGET_RATIO = 0.55`, `LOWERING_CEIL_SPEED_RATIO = 0.04` — ceiling descent config
- `LEVEL_HEIGHT_OVERRIDE` — per-level play area height overrides for short maps (level 7: 70%)
- `MAX_OBSTACLES = 8`, `MAX_BALLS = 64`
- `TRAINING` — MaskablePPO hyperparameters (n_envs=16, n_steps=8192, batch_size=8192, n_epochs=4, 1024×512×256 MLP, gamma=0.999, 300M total_timesteps)
- `RECURRENT_TRAINING` — RecurrentPPO hyperparameters (n_envs=16, n_steps=2048, 512×256 MLP + 256-unit LSTM, gamma=0.999, 300M total_timesteps)
- `CURRICULUM` — 6 phases: `(start_timestep, min_level, max_level, powerups_enabled)` — progressive level expansion
- `LEVEL_BACKGROUNDS` — per-level RGB background colors for renderer
- `compute_ball_properties(level, width, height)` — derives radius, max_yspeed, yacc from kinematic equations

### `src/engine.py` — `BubbleTroubleEngine`

Pure-numpy physics engine. Zero pygame dependency. All ball state lives in preallocated numpy arrays of shape `(MAX_BALLS=64,)`. Obstacle state in arrays of shape `(MAX_OBSTACLES=8,)`. Updates are vectorized where possible.

**Ball physics**: 6 ball levels (measured from reference video — all bounce to similar heights ~36-44% of play area, with longer periods for larger balls). Kinematic equations. Gravity derived from `bounce_height` and `bounce_time` per ball level: `yacc = 2 * bounce_height / (bounce_time/2)^2`. Balls bounce off walls (horizontal), floor (vertical, reset to `-max_yspeed * sqrt(bounciness)`), and obstacles (minimum-penetration-axis push). Balls that reach the ceiling (`y < 0`) are **destroyed outright** regardless of size — this is the ceiling pop mechanic. Each ball has a `bounciness` property (default 1.0, configurable per ball in level defs) that scales floor-bounce speed. Balls can have `flags`: `BALL_FLAG_STATIC` balls have zero horizontal speed (they bounce vertically in place). Static balls' children inherit `BALL_FLAG_NORMAL` on split. All spatial values scale proportionally to display dimensions.

**Ball splitting & pop physics**: Children receive a chain-depth-aware upward impulse plus inheritance of the parent's vertical velocity. Each ball tracks `ball_chain_depth` — how many consecutive splits happened in its ancestry without touching the floor. The chain resets to 0 on any floor bounce.

Formula: `child_yspeed = POP_VELOCITY_INHERIT * parent_yspeed - BASE * (1+CHAIN_GAIN)^child_depth / child_level^MASS_EXP`

Where `child_depth = parent.chain_depth + 1`. Key constants: `POP_VELOCITY_INHERIT = 0.37`, `POP_BASE_IMPULSE_RATIO = 0.52` (×height), `POP_CHAIN_GAIN = 0.22`, `POP_MASS_EXPONENT = 0.3`. Validated ceiling-pop constraints (parent popped near crest, from ~y=150):
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

**Power-ups**: 15% drop chance per ball pop. Two types: laser grid (next laser sticks at ceiling for 5s, firing perpendicular crosshairs), hourglass (adds 15 seconds to the level timer). Power-ups fall from the ball's pop position to the floor before becoming pickable. Only one power-up can exist at a time (checked via `powerup_on_ground or powerup_falling`).

**Level progression**: Sequential levels 1-12. When all balls are cleared, advance to next level. Steps reset per level (each level gets a full 60-second timer). Lasers, power-ups, and obstacles are cleared/reloaded on level transition.

**Pop effects tracking**: `recent_pops` list collects `(x, y, radius)` for balls destroyed/split each step. Included in `get_state()` for renderer star-burst effects. Cleared at start of each `step()`.

**`_finalize_info(info)`**: Helper that stamps cumulative episode counters (`shots_wasted`, `danger_splits`, `levels_cleared`, `highest_level`, `ceiling_pops`) onto the info dict. Called before every return in `step()` to ensure consistent data regardless of which code path exits.

**`step()` return paths** (4 total):
1. Already done → `(0, True, False, info)`
2. Agent-ball collision (death) → `(reward, True, False, info)`
3. All levels cleared → `(reward, False, True, info)`
4. Normal/timeout → `(reward, False, truncated, info)`

### `src/bubble_env.py` — `BubbleTroubleEnv(gymnasium.Env)`

Wraps the engine as a standard Gymnasium environment.

**Action space**: `MultiDiscrete([3, 2])` — movement (0=LEFT, 1=RIGHT, 2=STILL) × shooting (0=SHOOT, 1=NO_SHOOT). Allows simultaneous move + shoot.

**Observation space**: `Box(low=-1, high=1, shape=(704,), dtype=float32)`. All features normalized to [-1, 1]. No frame stacking — temporal context is encoded via `peak_height`, `intercept_x`, and velocity features.

Layout (704 elements):

| Index | Count | Feature |
|-------|-------|---------|
| 0-63 | 64 | Ball center x (sorted by horizontal distance to agent) |
| 64-127 | 64 | Ball center y |
| 128-191 | 64 | Ball x-speed |
| 192-255 | 64 | Ball y-speed |
| 256-319 | 64 | Ball radius |
| 320-383 | 64 | Ball level (normalized by MAX_BALL_LEVEL) |
| 384-447 | 64 | Ball is_active flag (1.0 if ball exists, 0.0 if empty slot) |
| 448-511 | 64 | Relative x: signed distance from agent to ball center (saves network subtraction) |
| 512-575 | 64 | Peak height: predicted apex y when rising, 1.0 when falling (ceiling pop candidates) |
| 576-639 | 64 | Intercept x: predicted ball center x when laser reaches ball's current height. Formula: ball_cx + ball_xspeed × (ball_cy / screen_height). Directly encodes where to stand to intercept each ball — a product across position, speed, and laser-speed constant that the MLP cannot derive without knowing the laser speed. |
| 640 | 1 | Agent center x |
| 641 | 1 | Laser active (1.0 / -1.0) |
| 642 | 1 | Longest laser length |
| 643 | 1 | Laser x position (where the laser is horizontally) |
| 644 | 1 | Can-fire (1.0 if SHOOT would fire, -1.0 if all slots busy) |
| 645 | 1 | Ball count ratio |
| 646 | 1 | Time remaining ratio |
| 647 | 1 | Current level (auto-scales with NUM_LEVELS) |
| 648 | 1 | Best chain intercept x: predicted intercept position of the ball with the best ceiling pop chain potential. Uses full pop physics model (velocity inheritance, chain-depth impulse, level-dependent gravity, kinematic prediction). Tells the agent WHERE to stand for the best chain shot. |
| 649 | 1 | Best chain quality: how good the best chain pop opportunity is right now. quality = max_rise/ball_cy weighted by level. Values near +1.0 = children WILL reach ceiling if ball is hit now. Values near -1.0 = no viable chain opportunity. Critical for levels 9-10 (lvl-5/6 balls). |
| 650 | 1 | Has laser grid (1.0 / -1.0) |
| 651 | 1 | Laser stuck at ceiling (1.0 / -1.0) |
| 652 | 1 | Power-up visible on ground or falling (1.0 / -1.0) |
| 653 | 1 | Signed distance to power-up |
| 654 | 1 | n_rising_ratio: fraction of active balls currently rising (yspeed<0). Pre-aggregates 64 yspeed features into one signal encoding how wide the current ceiling-pop window is. |
| 655 | 1 | closest_approach_time: seconds until nearest approaching ball reaches agent x. = |relative_x| / |xspeed|, capped 3s, normalized. Non-trivial division across feature channels; tells agent how urgently to dodge. |
| 656-703 | 48 | Obstacle features: 8 slots x 6 values (center_x, center_y, width, height, type, is_passable) |

**Action masking**: `action_masks()` returns flat bool array of shape `(5,)` for `MultiDiscrete([3, 2])`: `[LEFT, RIGHT, STILL, SHOOT, NO_SHOOT]`. SHOOT is masked when no laser slot is available. Used by `MaskablePPO` from sb3-contrib.

Ball slots are sorted by horizontal distance to agent center. Empty ball/obstacle slots are zero-filled.

**Normalization**: Ball yspeed and radius use **fixed normalizers** (`max_possible_yspeed` = 4× max ball level bounce speed to cover chain pop velocities, `max_possible_radius` from max ball level) rather than dynamic normalization. This ensures stationary observation statistics regardless of which balls are alive.

**`set_curriculum()`**: Called by the training callback to adjust `start_level`, `max_level`, and `enable_powerups` between episodes.

### `src/train_ppo.py` — MaskablePPO Training

Uses sb3-contrib `MaskablePPO` with `ActionMasker`. Default: 300M timesteps, cycling curriculum.

**Curriculum structure**:
1. Progressive expansion (35%): levels 1-3 → 1-5 → 3-8 → 5-10 → 7-12
2. Full game + hard refreshers (65%): full 1-12, then hard 5-12, then final 1-12

All envs always play the same level range per phase — coherent batches, no distribution shock. Hard refresher phases force practice on levels the agent rarely reaches organically.

**Training stack** (wrapper order matters):
1. `BubbleTroubleEnv` — base gymnasium env
2. `ActionMasker` — wraps env with `action_masks()` callback for MaskablePPO
3. `Monitor` — episode stats tracking
4. `SubprocVecEnv` / `DummyVecEnv` — parallelization (16 envs)
5. `VecNormalize(norm_obs=False, norm_reward=True)` — reward normalization only

**Key hyperparameters**: gamma=0.999 (~1000-step planning horizon for ceiling pop chains), target_kl guards policy update magnitude, 8 gradient epochs per rollout, 1024×512×256 MLP.

**Callbacks**:
- `CurriculumCallback` — advances through `CURRICULUM` phases in config
- `MetricsCallback` — logs game stats to TensorBoard at episode boundaries
- `EvalCallback` — periodic evaluation (every 500K steps), saves best model
- `CheckpointCallback` — snapshot every 5M steps

**Saved artifacts**: `final_model.zip` + `final_model_vecnorm.pkl`. VecNormalize stats needed for evaluation.

### `src/train_recurrent.py` — RecurrentPPO (LSTM) Training

Uses sb3-contrib `RecurrentPPO` with `MlpLstmPolicy`. Same curriculum as MaskablePPO but with an LSTM policy that learns temporal strategies natively.

**Why LSTM**: Multi-step strategies (position → wait → shoot for ceiling chain) are inherently sequential. An MLP treats each frame independently and relies on hand-crafted temporal features. An LSTM learns temporal patterns from raw experience — when to wait, when to shoot, level-specific dodge patterns over time.

**No action masking**: RecurrentPPO doesn't support `ActionMasker`. The LSTM learns to avoid invalid shots via the `can_fire` observation feature and `wasted_shot` penalty. No `ActionMasker` wrapper is used.

**Architecture**: 512×256 MLP → 256-unit LSTM (separate actor/critic LSTMs, `enable_critic_lstm=True`). Smaller MLP than MaskablePPO because the LSTM adds its own representational capacity.

**Training stack** (wrapper order matters):
1. `BubbleTroubleEnv` — base gymnasium env
2. `Monitor` — episode stats tracking (no ActionMasker)
3. `SubprocVecEnv` / `DummyVecEnv` — parallelization (16 envs)
4. `VecNormalize(norm_obs=False, norm_reward=True)` — reward normalization only

**Saved artifacts**: `checkpoints/recurrent/final_model.zip` + `final_model_vecnorm.pkl`.

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

## 12 Levels

Levels 1-12 are measured from the reference video (Bubble Struggle 2: Rebubbled).

| Level | Balls | Obstacles | Notes |
|-------|-------|-----------|-------|
| 1 | 1x lvl2 (yellow) | none | Single small ball |
| 2 | 1x lvl3 (green) | none | Single medium ball |
| 3 | 1x lvl4 (red) | none | Single medium-large ball |
| 4 | 2x lvl3 (orange) | none | Two balls |
| 5 | 1x lvl3 + 1x lvl4 | door wall | Door opens when left ball fully popped |
| 6 | 8x lvl1 (blue/purple) | lowering ceiling | Ceiling descends, compresses play area |
| 7 | 12x lvl1 (4 groups of 3) | short map (70%) + static ceiling | Raised floor |
| 8 | 1x lvl3 + 1x lvl4 + 1x lvl5 | 2 opening walls | Walls slide apart after sections cleared |
| 9 | 1x lvl5 (static) + 2x lvl4 | none | Red center + yellow sides |
| 10 | 1x lvl6 (dark red) | none | Single huge ball |
| 11 | 8x lvl2 (static, high bounce) | none | Two groups at top, bounce to 76% height |
| 12 | 6x lvl3 (static) + 1x lvl3 | none | Mixed purple/blue-purple, one mobile green |

## Reference Videos — `assets/game_levels/`

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
ffmpeg -i assets/game_levels/levelNN.mp4 -vf "select=eq(n\,0)" -vframes 1 -y /tmp/levelNN_frame0.png

# Gameplay frame at T seconds
ffmpeg -i assets/game_levels/levelNN.mp4 -ss T -vframes 1 -y /tmp/levelNN_tT.png

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
| `hit_ball_base` | 0.8 * level | Ball split (level 2-6), scaled by height_bonus_factor + shot_quality_bonus |
| `pop_ball` | 1.5 | Smallest ball (level 1) destroyed, scaled by height_bonus_factor + shot_quality_bonus |
| `height_bonus_factor` | 0.5 | Up to 50% extra reward for hitting balls near ceiling (spatial signal) |
| `shot_quality_scale` | 0.25 | Up to 0.25 × height_factor extra for hitting fast-rising balls (temporal signal). upward_factor = min(1, \|yspeed\|/max_yspeed) when ball is rising, 0 otherwise. Teaches the agent to shoot on the upswing not at apex — fast-rising parents produce faster children for chain pops. |
| `clutter_pop_bonus` | 0.10/ball | Extra per existing lvl-1 ball beyond 2 when popping — rewards clearing crowded screens |
| `clutter_split_penalty` | -0.04/ball | Penalty per lower-level ball beyond 3 when splitting lvl≥3 — discourages adding more chaos to cluttered screens |
| `danger_split_penalty` | -0.2 | Splitting a ball that's close to agent and low on screen |
| `finish_level` | 10.0 × (1 + (level-1) × 0.25) | All balls cleared — scaled by level (L1: 10.0, L6: 22.5, L12: 37.5) |
| `time_bonus_scale` | 8.0 * (remaining/max) | Bonus for faster clears |
| `timeout` | -3.0 | Level timer runs out without clearing |
| `game_over` | -8.0 × (1 + levels_cleared × 0.3) | Agent hit by ball — scaled by progress (0 cleared: -8.0, 5 cleared: -20.0, 11 cleared: -34.4) |
| `pickup_powerup` | 0.3 | Agent picks up a ground power-up |
| `clear_all_levels` | 50.0 | Beat all levels |

**Ceiling pop rewards**: When a ball hits the ceiling (y < 0), it's destroyed outright with no split. The reward equals the full subtree value of manually clearing all descendants: `CEILING_POP_VALUES[level]` (L1:1.5, L2:4.6, L3:11.6, L4:26.4, L5:56.8, L6:118.4). This makes ceiling pops strategically equivalent to manual clearing but better via time/risk savings.

**Height bonus**: `height_factor = 1 + 0.5 * max(0, 1 - ball_cy / effective_height)` — up to 50% extra reward for hitting balls near the ceiling, guiding the agent toward ceiling pop setups.

## Curriculum Learning

Cycling curriculum defined in `config.py:CURRICULUM` (300M total steps). Two-stage design: progressive expansion then full-game mastery with hard refreshers.

**Stage 1 — Progressive expansion** (first ~105M steps):

| Phase | Timestep | Levels | Power-ups | Focus |
|-------|----------|--------|-----------|-------|
| 0 | 0 | 1-3 | Off | Learn to shoot (simple balls) |
| 1 | 20M | 1-5 | Off | Add obstacle levels (door wall) |
| 2 | 50M | 3-8 | Off | Opening walls, static balls, larger balls |
| 3 | 80M | 5-10 | Off | Level-6 ball — discover ceiling pop chains |
| 4 | 105M | 7-12 | Off | Hard late levels |

**Stage 2 — Full game + hard refreshers** (remaining ~195M steps):

| Phase | Timestep | Levels | Power-ups | Focus |
|-------|----------|--------|-----------|-------|
| 5 | 130M | 1-12 | On | Full game with power-ups |
| 6 | 180M | 5-12 | On | Hard refresher — force late-level practice |
| 7 | 240M | 1-12 | On | Final full game |

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
- Observation space: 110 → 206 → 222 → 224 → 704 elements (MAX_OBS_BALLS 16 → 64: agent now sees all balls from full level-6 split tree)
- Ball levels: 4 → 6 (fixed normalizers now use level-6 values)
- Ball colors reordered: lvl1=blue, lvl2=yellow, lvl3=green (measured from reference video)
- Pop physics: chain-depth-aware model (BASE=0.52×h, CHAIN_GAIN=0.22, MASS_EXP=0.3); `ball_chain_depth` array added; clamp raised to 4×
- Ceiling pop rewards: balls destroyed at ceiling get full subtree value (CEILING_POP_VALUES)
- Height bonus: up to 50% extra reward for hitting balls near ceiling
- MAX_BALLS: 32 → 64
- NUM_LEVELS: 7 → 12
- Levels 1-12 redefined from reference video measurements
- Ball flags system added (BALL_FLAG_STATIC for non-moving balls)
- Obstacle type system added (OBSTACLE_DOOR, OBSTACLE_OPENING, OBSTACLE_LOWERING_CEIL)
- Short map system added (LEVEL_HEIGHT_OVERRIDE for raised floor levels)
- Dynamic obstacles: opening walls slide apart, lowering ceiling descends over time
- Training: 500M → 300M steps, two-phase (warmup + per-level), VecNormalize stats transfer
- All previously trained models are incompatible
