# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workflow Orchestration

### 1. Plan Node Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately — don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: knowing everything I know now, implement the elegant solution
- Skip this for simple, obvious fixes — don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests — then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management
1. **Plan First**: Write plan to `tasks/TODO.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/TODO.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

## Core Principles
- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.

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

# Human demo recording (for BC pretraining)
python src/record_demos.py              # record from level 1
python src/record_demos.py 5            # record from level 5
python src/combine_demos.py checkpoints/user_warmups/demo_*.npz  # merge demo files

# BC pretraining → PPO warm start → DAPG fine-tuning
python src/pretrain_bc.py checkpoints/user_warmups/demo_*.npz
python src/train_ppo.py --warmup checkpoints/user_warmups/bc_pretrained.zip \
    --demo checkpoints/user_warmups/demo_*.npz

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

> Full architecture reference: [`docs/architecture.md`](docs/architecture.md)

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
| `game_over` | -8.0 × (1 + levels_cleared × 0.5) | Agent hit by ball — scaled by progress (0 cleared: -8.0, 5 cleared: -28.0, 11 cleared: -52.0) |
| `pickup_powerup` | 0.3 | Agent picks up a ground power-up |
| `clear_all_levels` | 50.0 | Beat all levels |

**Ceiling pop rewards**: When a ball hits the ceiling (y < 0), it's destroyed outright with no split. The reward equals the full subtree value of manually clearing all descendants: `CEILING_POP_VALUES[level]` (L1:1.5, L2:4.6, L3:11.6, L4:26.4, L5:56.8, L6:118.4). This makes ceiling pops strategically equivalent to manual clearing but better via time/risk savings.

**Height bonus**: `height_factor = 1 + 0.5 * max(0, 1 - ball_cy / effective_height)` — up to 50% extra reward for hitting balls near the ceiling, guiding the agent toward ceiling pop setups.

## Curriculum Learning

Cycling curriculum defined in `config.py:CURRICULUM` (1B total steps for PPO, 120M for RecurrentPPO). Multi-phase design: foundation → complexity → hard introduction → full integration → mastery cycling → final polish.

| Phase | Timestep | Levels | Power-ups | Focus |
|-------|----------|--------|-----------|-------|
| 0 | 0 | 1-5 | Off | Foundation: shooting, dodging, door wall (80M steps) |
| 1 | 80M | 3-8 | Off | Complexity: opening walls, static balls, lowering ceiling (120M) |
| 2 | 200M | 5-10 | On | Hard intro: first L9-10 exposure with scaffold (60M) |
| 3 | 260M | 7-12 | On | Hard intro: first L11-12 exposure (60M) |
| 4 | 320M | 9-12 | On | Focused hard-level training (80M) |
| 5 | 400M | 1-12 | On | Full sequential 1-12 integration (140M) |
| 6 | 540M | 9-12 | On | Hard refresher L9-12 (60M) |
| 7 | 600M | 1-12 | On | Full game (40M) |
| 8 | 640M | 10-12 | On | Hard refresher L10-12 (60M) |
| 9 | 700M | 1-12 | On | Full game (40M) |
| 10 | 740M | 11-12 | On | Hard refresher L11-12 (60M) |
| 11 | 800M | 1-12 | On | Full game (40M) |
| 12 | 840M | 12-12 | On | L12 mastery (50M) |
| 13 | 890M | 1-12 | On | Full game consolidation (60M) |
| 14 | 950M | 10-12 | On | Final hard push (50M) |

## Known Patterns & Pitfalls

> Full pitfalls & breaking changes list: [`docs/pitfalls.md`](docs/pitfalls.md)
