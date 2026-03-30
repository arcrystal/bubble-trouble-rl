"""Record human demonstrations for behavioral cloning pretraining.

Plays the game level-by-level with an interactive timeline scrubber.
After each attempt (win or lose), the scrubber lets you rewind to any
frame and resume playing from there — enabling "perfect play" recordings
by retrying tricky sections while keeping good earlier play.

Engine state is deep-copied every frame (~20KB per snapshot, ~60MB for
a 60-second level at 60fps). Memory is freed between levels.

Controls during gameplay:
    Left/Right arrows — move
    Space — shoot (one press = one shot)
    R — restart current level from scratch

Scrubber controls (appears after each attempt):
    Left/Right — scrub timeline (hold to accelerate)
    Click/drag — scrub timeline via mouse
    Enter — resume playing from current frame
    S — save level recording & advance (only when level completed)
    R — restart level from scratch
    Q — save all recorded data & quit

Output: checkpoints/user_warmups/demo_TIMESTAMP.npz
    observations  (T, 968) float32 — normalized observation at each step
    actions       (T, 2)   int64   — [movement, shooting] per step
    level_ids     (T,)     int32   — which level each step belongs to
    rewards       (T,)     float32 — reward received at each step (for value pretraining)

Usage:
    python src/record_demos.py               # start from level 1
    python src/record_demos.py 5             # start from level 5
"""

import sys
import os
import time
import copy

import numpy as np
import pygame

from bubble_env import BubbleTroubleEnv
from config import NUM_LEVELS, DEFAULT_FPS


# UI colors
_WHITE = (255, 255, 255)
_GREEN = (100, 255, 100)
_RED = (255, 100, 100)
_GRAY = (180, 180, 180)


def _render_frame(renderer, engine):
    """Render an engine snapshot without clock.tick or display.update.

    Returns the window surface so the caller can draw overlays before
    flushing the display.
    """
    state = engine.get_state()
    renderer._pop_effects.clear()  # no stale pop animations while scrubbing
    surface = renderer._draw(state)
    renderer.window.blit(surface, (0, 0))
    return renderer.window


def _show_scrubber(env, snapshots, level_obs, level, completed):
    """Interactive timeline scrubber. Returns (choice, frame_index).

    Choices: "resume", "save", "restart", "quit".
    """
    renderer = env._renderer
    if renderer is None or not snapshots:
        return "restart", 0

    window = renderer.window
    sw, sh = window.get_size()
    n = len(snapshots)
    pos = n - 1  # start at end

    font_title = pygame.font.SysFont(None, 36)
    font = pygame.font.SysFont(None, 22)
    clock = pygame.time.Clock()
    hold = 0

    fill_color = _GREEN if completed else _RED
    title = f"Level {level} {'Complete!' if completed else '— Failed'}"
    title_color = _GREEN if completed else _RED

    BAR_H = 76
    MARGIN = 30
    bar_y = sh - BAR_H
    track_x, track_w = MARGIN, sw - 2 * MARGIN
    track_y = bar_y + 36
    TRACK_H = 6
    HANDLE_R = 8

    while True:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit", pos
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    return "resume", pos
                if event.key == pygame.K_s and completed:
                    return "save", pos
                if event.key == pygame.K_r:
                    return "restart", pos
                if event.key == pygame.K_q:
                    return "quit", pos

        # Keyboard scrubbing with acceleration
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_RIGHT]:
            hold += 1
            speed = min(1 + hold // 3, 50)
            if keys[pygame.K_LEFT]:
                pos = max(0, pos - speed)
            else:
                pos = min(n - 1, pos + speed)
        else:
            hold = 0

        # Mouse scrubbing (click/drag on timeline bar)
        if pygame.mouse.get_pressed()[0]:
            mx, my = pygame.mouse.get_pos()
            if bar_y <= my <= bar_y + BAR_H and track_x <= mx <= track_x + track_w:
                frac = (mx - track_x) / max(track_w, 1)
                pos = max(0, min(n - 1, round(frac * (n - 1))))

        # Render game state at scrubber position (no clock/flip inside)
        env.engine = snapshots[pos]
        _render_frame(renderer, snapshots[pos])

        # --- Overlay bar ---
        overlay = pygame.Surface((sw, BAR_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 210))
        window.blit(overlay, (0, bar_y))

        # Title
        window.blit(font_title.render(title, True, title_color), (MARGIN, bar_y + 6))

        # Frame counter
        ftxt = font.render(f"Frame {pos} / {n - 1}", True, _WHITE)
        window.blit(ftxt, ftxt.get_rect(right=sw - MARGIN, top=bar_y + 12))

        # Track
        pygame.draw.rect(window, (80, 80, 80), (track_x, track_y, track_w, TRACK_H))
        frac = pos / max(n - 1, 1)
        fw = int(frac * track_w)
        pygame.draw.rect(window, fill_color, (track_x, track_y, fw, TRACK_H))
        pygame.draw.circle(window, _WHITE, (track_x + fw, track_y + TRACK_H // 2), HANDLE_R)

        # Instructions
        parts = ["LEFT/RIGHT scrub", "ENTER resume"]
        if completed:
            parts.append("S save")
        parts += ["R restart", "Q quit"]
        inst = font.render("    ".join(parts), True, _GRAY)
        window.blit(inst, inst.get_rect(centerx=sw // 2, top=track_y + TRACK_H + 10))

        pygame.display.update()


def main():
    save_dir = "checkpoints/user_warmups"
    os.makedirs(save_dir, exist_ok=True)

    start_level = 1
    if len(sys.argv) > 1:
        start_level = max(1, min(int(sys.argv[1]), NUM_LEVELS))

    env = BubbleTroubleEnv(
        render_mode="human",
        start_level=start_level,
        max_level=start_level,
        enable_powerups=True,
        sequential_levels=True,
    )

    all_obs = []
    all_actions = []
    all_level_ids = []
    all_rewards = []
    saved_levels = []
    quit_requested = False

    current_level = start_level

    print(f"\n{'='*55}")
    print(f"  Demo Recording — Starting from Level {start_level}")
    print(f"  Gameplay:  arrows=move  space=shoot  R=restart")
    print(f"  Scrubber appears after each attempt (win or lose)")
    print(f"{'='*55}\n")

    while current_level <= NUM_LEVELS and not quit_requested:
        env.set_curriculum(
            start_level=current_level,
            max_level=current_level,
            enable_powerups=True,
        )
        obs, info = env.reset()
        env.render()

        snapshots = []
        level_obs = []
        level_actions = []
        level_rewards = []

        while True:  # play / scrub loop for this level
            # --- PLAY PHASE ---
            done = False

            while not done:
                # env.render() handles clock.tick internally — no separate clock needed
                shoot_this_frame = False

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit_requested = True
                        done = True
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            shoot_this_frame = True
                        elif event.key == pygame.K_r:
                            obs, info = env.reset()
                            env.render()
                            snapshots.clear()
                            level_obs.clear()
                            level_actions.clear()
                            level_rewards.clear()

                if quit_requested or done:
                    break

                keys = pygame.key.get_pressed()
                move = 0 if keys[pygame.K_LEFT] else (1 if keys[pygame.K_RIGHT] else 2)
                shoot = 0 if shoot_this_frame else 1
                action = np.array([move, shoot])

                # Snapshot engine BEFORE action (for rewind)
                snapshots.append(copy.deepcopy(env.engine))
                level_obs.append(obs.copy())
                level_actions.append(action.copy())

                obs, reward, terminated, truncated, info = env.step(action)
                level_rewards.append(reward)
                env.render()
                done = terminated or truncated

            if quit_requested:
                break

            completed = (
                info.get("levels_cleared", 0) > 0
                or info.get("game_cleared", False)
            )

            if not snapshots:
                break

            # --- SCRUBBER PHASE ---
            choice, frame = _show_scrubber(
                env, snapshots, level_obs, current_level, completed
            )

            if choice == "resume":
                # Restore engine to selected frame and continue playing
                env.engine = copy.deepcopy(snapshots[frame])
                obs = level_obs[frame].copy()
                snapshots = snapshots[:frame]
                level_obs = level_obs[:frame]
                level_actions = level_actions[:frame]
                level_rewards = level_rewards[:frame]
                if env._renderer:
                    env._renderer._pop_effects.clear()
                env.render()
                # loops back to play phase

            elif choice == "save":
                all_obs.extend(level_obs)
                all_actions.extend(level_actions)
                all_rewards.extend(level_rewards)
                all_level_ids.extend([current_level] * len(level_actions))
                saved_levels.append(current_level)
                print(f"  Level {current_level}: saved ({len(level_actions):,} steps)")
                current_level += 1
                break

            elif choice == "restart":
                obs, info = env.reset()
                env.render()
                snapshots.clear()
                level_obs.clear()
                level_actions.clear()
                level_rewards.clear()
                # loops back to play phase

            elif choice == "quit":
                quit_requested = True
                break

    env.close()

    if all_obs:
        obs_arr = np.array(all_obs, dtype=np.float32)
        act_arr = np.array(all_actions, dtype=np.int64)
        lvl_arr = np.array(all_level_ids, dtype=np.int32)
        rew_arr = np.array(all_rewards, dtype=np.float32)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        lvl_tag = "_".join(str(l) for l in saved_levels)
        path = os.path.join(save_dir, f"demo_lvl_{lvl_tag}_{timestamp}.npz")
        np.savez_compressed(
            path,
            observations=obs_arr,
            actions=act_arr,
            level_ids=lvl_arr,
            rewards=rew_arr,
        )
        print(f"\nSaved {len(act_arr):,} demo steps "
              f"(levels {', '.join(str(l) for l in saved_levels)}) -> {path}")
    else:
        print("\nNo data recorded.")


if __name__ == "__main__":
    main()
