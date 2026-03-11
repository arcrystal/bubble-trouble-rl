# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bubble Trouble arcade game clone with reinforcement learning training. The player controls an agent that shoots a vertical laser to pop bouncing balls that split into smaller balls (4→3→2→1→gone).

## Branches

- **rl-v2** — current active branch. Headless numpy engine + SB3 PPO training.
- **main** — original pygame RL env with Ray RLlib (legacy)
- **webapp** — browser-playable version via pygbag
- **gym** — older OpenAI Gym wrapper (legacy)

## Setup & Running

```bash
# Python 3.13 required
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train
python train.py                              # default: 10M steps, 8 envs
python train.py --timesteps 50000000         # custom duration
python train.py --resume checkpoints/final_model  # resume training

# Evaluate trained model
python evaluate.py play checkpoints/best/best_model.zip
python evaluate.py play model.zip --no-render   # headless stats
python evaluate.py benchmark                    # raw env speed test

# TensorBoard
tensorboard --logdir ./logs
```

## Architecture (rl-v2 branch)

Two-layer design: headless physics engine + gymnasium wrapper.

**`config.py`** — All constants, physics parameters, reward values, level definitions, training hyperparameters, curriculum phases. Single source of truth.

**`engine.py`** — `BubbleTroubleEngine`: pure-numpy physics with zero pygame dependency.
- Ball state in preallocated numpy arrays (`ball_x`, `ball_y`, etc.) shape `(MAX_BALLS,)`
- Vectorized updates: all balls updated in single numpy operations
- AABB-circle collision (agent-ball), line-circle collision (laser-ball)
- Power-ups: double harpoon (2 lasers), force field (survive 1 hit), hourglass (slow balls)
- Sequential level progression (1→7) with per-level time limits
- No `_will_collide()` — the expensive simulation from old code is eliminated

**`bubble_env.py`** — `BubbleTroubleEnv(gymnasium.Env)`: wraps engine with 107-element observation.
- Obs layout: 16 ball slots (sorted by distance) × 6 features + 3 agent + 2 global + 6 power-up
- All values normalized to [-1, 1]
- `set_curriculum()` method for adjusting difficulty during training
- Renderer is lazy-loaded only when `render_mode` is set

**`train.py`** — SB3 PPO training with `SubprocVecEnv`.
- `CurriculumCallback`: auto-advances difficulty (easy balls → full game + powerups)
- `MetricsCallback`: logs game-specific stats to TensorBoard
- CLI args for timesteps, resume, device, seed

**`renderer.py`** — Optional pygame-ce renderer. Reads engine state dict, draws everything. Only imported by `bubble_env.py` when rendering.

**`evaluate.py`** — Load checkpoint and run visual episodes or benchmark env speed.

**Legacy pygame files** (kept for reference/human play):
- `game.py`, `agent.py`, `AbstractBall.py`, `ball.py`, `laser.py`, `levels.py`, `direction.py`, `main.py`

## Key Design Decisions

- All spatial values scale proportionally to display dimensions (not hardcoded pixels)
- Ball physics: `calculate_vertical_motion()` in `config.py` derives gravity/speed from bounce height + time
- Ball removal uses swap-with-last for O(1) deletion from numpy arrays
- Rewards are dense and well-scaled (all within 1 order of magnitude)
- Action space: `Discrete(4)` — 0=LEFT, 1=RIGHT, 2=SHOOT, 3=STILL
- Power-ups are passive (auto-pickup on contact) — no action space expansion needed
