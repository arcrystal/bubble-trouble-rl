"""Pygame-based renderer for the headless Bubble Trouble engine.

This module is only imported when render_mode is set — it is NOT needed for training.
"""

import numpy as np
import pygame
from config import (
    POWERUP_LASER_GRID, POWERUP_HOURGLASS,
    DEFAULT_FPS, LEVEL_BACKGROUNDS, NUM_LEVELS,
    OBSTACLE_DOOR, OBSTACLE_LOWERING_CEIL, OBSTACLE_OPENING,
    BLACK, WHITE, RED, YELLOW, GREEN, BLUE, ORANGE, PURPLE,
    CYAN, GRAY, DARK_GRAY, DARK_RED, CRIMSON,
)

# Renderer-only UI colors
STONE          = (90,  75,  55)
LASER_COLOR    = (200, 200, 200)
DOOR_COLOR     = (75,  65,  50)   # Slightly different shade from stone
DOOR_BORDER    = (55,  50,  40)
DOOR_OPEN_COLOR  = (50, 45, 35)   # Dimmer shade for open door (passable by agent)
DOOR_OPEN_BORDER = (40, 35, 28)

POWERUP_COLORS = {
    POWERUP_LASER_GRID: CYAN,
    POWERUP_HOURGLASS: PURPLE,
}

LASER_GRID_SPACING = 18   # pixels between perpendicular grid lines
LASER_GRID_HALF_W  = 6    # half-width of the grid background and crosshairs
LASER_GRID_BG      = (60, 60, 60, 100)  # faint grey background (alpha blended)


# HUD layout
HUD_HEIGHT = 28  # Bottom score bar height


class PygameRenderer:
    """Renders engine state using pygame."""

    def __init__(self, width, height, fps=DEFAULT_FPS, speed=1.0):
        self.width = width
        self.height = height
        self.fps = fps
        self.speed = speed

        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((width, height + HUD_HEIGHT))
        pygame.display.set_caption("Bubble Trouble RL")
        self.clock = pygame.time.Clock()

        self.font = pygame.font.Font(None, 22)
        self.big_font = pygame.font.Font(None, 36)
        self.hud_font = pygame.font.Font(None, 20)

        # Pre-allocated draw surface (reused every frame)
        self._surface = pygame.Surface((width, height + HUD_HEIGHT))

        # Pre-rendered static HUD text
        self._p1_txt = self.hud_font.render("PLAYER 1", True, WHITE)
        self._p2_txt = self.hud_font.render("PLAYER 2", True, WHITE)

        # Pre-rendered "Get ready" overlay
        self._ready_overlay = pygame.Surface((280, 50), pygame.SRCALPHA)
        self._ready_overlay.fill((0, 0, 0, 160))
        self._ready_txt = self.big_font.render("Get ready", True, YELLOW)

        # Pre-allocated grid laser surface (reused when a stuck laser is active)
        self._grid_surf = pygame.Surface(
            (LASER_GRID_HALF_W * 2 + 1, height + 1), pygame.SRCALPHA)

        # Cached HUD text (re-rendered only when the value changes)
        self._cached_hud = {}  # key → (value, surface)

        # Pop effect state: list of (x, y, radius, frames_remaining)
        self._pop_effects = []
        self._agent_prev_x = None  # track previous x to select directional sprite

        # Load agent sprites if available, fall back to rectangle
        self.agent_sprites = {}
        try:
            for name in ("left", "still", "right"):
                img = pygame.image.load(f"assets/{name}.png")
                self.agent_sprites[name] = img
        except (pygame.error, FileNotFoundError):
            self.agent_sprites = None

        # Pre-scale agent sprites to final dimensions
        if self.agent_sprites:
            self._scaled_sprites = {}
        else:
            self._scaled_sprites = None

    def render(self, state):
        """Render the game state to the display window."""
        surface = self._draw(state)
        self.window.blit(surface, (0, 0))
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(int(self.fps * self.speed))

    def render_to_array(self, state):
        """Render to a numpy RGB array."""
        surface = self._draw(state)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2)
        )

    def _hud_text(self, key, text, font, color):
        """Return a cached font surface, re-rendering only when text changes."""
        cached = self._cached_hud.get(key)
        if cached is not None and cached[0] == text:
            return cached[1]
        surf = font.render(text, True, color)
        self._cached_hud[key] = (text, surf)
        return surf

    def _draw(self, state):
        """Draw all game elements to a surface."""
        surface = self._surface

        # Per-level background
        level = state.get("current_level", 1)
        bg_color = LEVEL_BACKGROUNDS.get(level, BLACK)
        surface.fill(bg_color)

        # Effective height (short map levels have raised floor)
        eff_h = state.get("effective_height", self.height)

        # Short map: draw stone floor at effective_height
        if eff_h < self.height:
            floor_y = int(eff_h)
            pygame.draw.rect(surface, STONE, (0, floor_y, self.width, self.height - floor_y))
            pygame.draw.line(surface, DARK_GRAY, (0, floor_y), (self.width, floor_y), 2)

        # --- "Get ready" splash for first 90 frames ---
        steps = state.get("steps", 999)

        # --- Draw obstacles ---
        n_obs = state.get("n_obstacles", 0)
        obs_types = state.get("obstacle_type", [])
        obs_timers = state.get("obstacle_timer", [])
        for i in range(n_obs):
            ox = int(state["obstacle_x"][i])
            oy = int(state["obstacle_y"][i])
            ow = int(state["obstacle_w"][i])
            oh = int(state["obstacle_h"][i])
            otype = int(obs_types[i]) if i < len(obs_types) else 0

            if otype == OBSTACLE_DOOR:
                # Door wall: closed = solid shade, open = dimmer shade (agent can pass)
                is_open = i < len(obs_timers) and obs_timers[i] < 0
                if is_open:
                    pygame.draw.rect(surface, DOOR_OPEN_COLOR, (ox, oy, ow, oh))
                    pygame.draw.rect(surface, DOOR_OPEN_BORDER, (ox, oy, ow, oh), 1)
                else:
                    pygame.draw.rect(surface, DOOR_COLOR, (ox, oy, ow, oh))
                    pygame.draw.rect(surface, DOOR_BORDER, (ox, oy, ow, oh), 1)
            elif otype == OBSTACLE_LOWERING_CEIL:
                # Lowering ceiling: dark red bar descending from top
                if oh > 0:
                    pygame.draw.rect(surface, DARK_RED, (ox, oy, ow, int(oh)))
                    pygame.draw.rect(surface, CRIMSON, (ox, oy, ow, int(oh)), 2)
            elif otype == OBSTACLE_OPENING:
                # Opening wall: stone wall (shrinking as it opens)
                if oh > 1:
                    pygame.draw.rect(surface, STONE, (ox, oy, ow, int(oh)))
                    pygame.draw.rect(surface, DARK_GRAY, (ox, oy, ow, int(oh)), 1)
            else:
                pygame.draw.rect(surface, STONE, (ox, oy, ow, oh))
                pygame.draw.rect(surface, DARK_GRAY, (ox, oy, ow, oh), 1)

        # Draw lasers
        laser_stuck = state.get("laser_stuck", [False] * len(state["laser_active"]))
        for i in range(len(state["laser_active"])):
            if state["laser_active"][i]:
                lx = int(state["laser_x"][i])
                ll = state["laser_length"][i]
                floor_y = int(eff_h)
                top_y = int(eff_h - ll)
                if laser_stuck[i] and ll > 0:
                    # Stuck grid laser: normal white line while travelling,
                    # grid pattern once it reaches the ceiling.
                    # Use a subsurface of the pre-allocated grid surface.
                    grid_h = int(ll) + 1
                    grid_w = LASER_GRID_HALF_W * 2 + 1
                    grid_surf = self._grid_surf.subsurface((0, 0, grid_w, grid_h))
                    grid_surf.fill((60, 60, 60, 80))
                    # Crosshairs at regular intervals (measured from ceiling downward)
                    for gy in range(0, int(ll), LASER_GRID_SPACING):
                        pygame.draw.line(grid_surf, LASER_COLOR,
                                         (0, gy), (LASER_GRID_HALF_W * 2, gy), 1)
                    # Main vertical line through center
                    pygame.draw.line(grid_surf, LASER_COLOR,
                                     (LASER_GRID_HALF_W, 0), (LASER_GRID_HALF_W, int(ll)), 1)
                    surface.blit(grid_surf, (lx - LASER_GRID_HALF_W, top_y))
                else:
                    pygame.draw.line(surface, LASER_COLOR, (lx, floor_y), (lx, top_y), 1)

        # Draw agent
        ax = state["agent_x"]
        aw = state["agent_w"]
        ah = state["agent_h"]
        agent_rect = pygame.Rect(ax, int(eff_h) - ah, aw, ah)

        if self.agent_sprites:
            # Cache scaled sprites on first use (dimensions are fixed)
            if not self._scaled_sprites:
                for sname in ("left", "still", "right"):
                    self._scaled_sprites[sname] = pygame.transform.scale(
                        self.agent_sprites[sname], (aw, ah))
            if self._agent_prev_x is not None and ax < self._agent_prev_x - 0.1:
                sprite_name = "left"
            elif self._agent_prev_x is not None and ax > self._agent_prev_x + 0.1:
                sprite_name = "right"
            else:
                sprite_name = "still"
            self._agent_prev_x = ax
            surface.blit(self._scaled_sprites[sprite_name], agent_rect)
        else:
            pygame.draw.rect(surface, WHITE, agent_rect)

        # Draw balls
        for i in range(state["n_balls"]):
            cx = int(state["ball_x"][i] + state["ball_radius"][i])
            cy = int(state["ball_y"][i] + state["ball_radius"][i])
            r = int(state["ball_radius"][i])
            color = tuple(state["ball_color"][i])
            pygame.draw.circle(surface, color, (cx, cy), r)
            pygame.draw.circle(surface, WHITE, (cx, cy), r, 1)  # thin outline on all balls

        # Draw ground power-up
        if state.get("powerup_on_ground"):
            px = int(state["powerup_ground_x"])
            py = int(state["powerup_ground_y"])
            ptype = state["powerup_ground_type"]
            if ptype == POWERUP_HOURGLASS:
                # Clock: white circle face with hour and minute hands
                r = 10
                pygame.draw.circle(surface, WHITE, (px, py), r)
                pygame.draw.circle(surface, BLACK, (px, py), r, 1)
                # Minute hand (pointing up, 12 o'clock)
                pygame.draw.line(surface, BLACK, (px, py), (px, py - r + 3), 1)
                # Hour hand (pointing right, 3 o'clock)
                pygame.draw.line(surface, BLACK, (px, py), (px + r - 4, py), 1)
                # Small center dot
                pygame.draw.circle(surface, BLACK, (px, py), 1)
            elif ptype == POWERUP_LASER_GRID:
                # Grid icon: small cyan rectangle with a grid pattern
                r = 9
                rect = pygame.Rect(px - r, py - r, r * 2, r * 2)
                pygame.draw.rect(surface, (20, 20, 20), rect)
                # Draw mini grid lines
                for gx in range(px - r, px + r + 1, 6):
                    pygame.draw.line(surface, CYAN, (gx, py - r), (gx, py + r), 1)
                for gy in range(py - r, py + r + 1, 6):
                    pygame.draw.line(surface, CYAN, (px - r, gy), (px + r, gy), 1)
                pygame.draw.rect(surface, CYAN, rect, 1)
            else:
                color = POWERUP_COLORS.get(ptype, WHITE)
                pygame.draw.circle(surface, color, (px, py), 8)

        # --- Pop effects ---
        # Add new pops from state
        for (px, py, pr) in state.get("recent_pops", []):
            self._pop_effects.append((px, py, pr, 10))  # 10 frames duration

        # Draw and decay existing pop effects
        remaining = []
        for (px, py, pr, frames) in self._pop_effects:
            alpha = frames / 10.0
            color_val = int(255 * alpha)
            star_color = (color_val, color_val, min(255, int(color_val * 0.6)))
            size = int(pr * 1.5 * alpha)
            if size > 1:
                # Simple star-burst: 4 lines radiating from center
                ipx, ipy = int(px), int(py)
                pygame.draw.line(surface, star_color, (ipx - size, ipy), (ipx + size, ipy), 1)
                pygame.draw.line(surface, star_color, (ipx, ipy - size), (ipx, ipy + size), 1)
                pygame.draw.line(surface, star_color, (ipx - size//2, ipy - size//2),
                                 (ipx + size//2, ipy + size//2), 1)
                pygame.draw.line(surface, star_color, (ipx + size//2, ipy - size//2),
                                 (ipx - size//2, ipy + size//2), 1)
            if frames > 1:
                remaining.append((px, py, pr, frames - 1))
        self._pop_effects = remaining

        is_infinity = state.get("infinity_mode", False)

        if is_infinity:
            # --- Infinity: difficulty bar (at bottom of play area) ---
            difficulty = state.get("difficulty", 0.0)
            bar_y = self.height - 4
            bar_width = self.width
            pygame.draw.rect(surface, DARK_GRAY, (0, bar_y, bar_width, 4))
            # Color ramps from green → yellow → red as difficulty increases
            r = int(min(255, 510 * difficulty))
            g = int(min(255, 510 * (1.0 - difficulty)))
            pygame.draw.rect(surface, (r, g, 0), (0, bar_y, int(bar_width * difficulty), 4))

            # --- Infinity HUD ---
            hud_y = self.height
            pygame.draw.rect(surface, BLACK, (0, hud_y, self.width, HUD_HEIGHT))

            # Survival time (left)
            elapsed = state.get("elapsed_time", 0.0)
            mins = int(elapsed) // 60
            secs = int(elapsed) % 60
            time_txt = self._hud_text("inf_time", f"TIME  {mins}:{secs:02d}", self.hud_font, WHITE)
            surface.blit(time_txt, (8, hud_y + 6))

            # Ball count (right)
            n_balls = state.get("n_balls", 0)
            balls_txt = self._hud_text("inf_balls", f"BALLS  {n_balls}", self.hud_font, WHITE)
            surface.blit(balls_txt, (self.width - balls_txt.get_width() - 8, hud_y + 6))
        else:
            # --- Regular mode: timer bar ---
            if state["max_steps"] > 0:
                time_remaining = max(0, state["max_steps"] - state["steps"])
                ratio = time_remaining / state["max_steps"]
                bar_y = self.height - 4
                bar_width = self.width
                pygame.draw.rect(surface, DARK_GRAY, (0, bar_y, bar_width, 4))
                bar_color = RED
                pygame.draw.rect(surface, bar_color, (0, bar_y, int(bar_width * ratio), 4))

            # --- Regular HUD score bar ---
            hud_y = self.height
            pygame.draw.rect(surface, BLACK, (0, hud_y, self.width, HUD_HEIGHT))

            surface.blit(self._p1_txt, (8, hud_y + 6))

            level_txt = self._hud_text("reg_level", f"LEVEL {level}", self.hud_font, RED)
            level_rect = level_txt.get_rect(center=(self.width // 2, hud_y + HUD_HEIGHT // 2))
            surface.blit(level_txt, level_rect)

            surface.blit(self._p2_txt, (self.width - self._p2_txt.get_width() - 8, hud_y + 6))

        # Active power-up indicators (top right of play area)
        indicators = []
        if state.get("has_laser_grid"):
            indicators.append(("Grid", CYAN))
        stuck_timers = state.get("laser_stuck_timer", [])
        for t in stuck_timers:
            if t > 0:
                indicators.append((f"Grid {t:.1f}s", CYAN))

        for i, (label, color) in enumerate(indicators):
            txt = self._hud_text(f"indicator_{i}", label, self.font, color)
            surface.blit(txt, (self.width - 80, 5 + i * 18))

        # --- "Get ready" splash ---
        if steps < 90:
            ox = (self.width - 280) // 2
            oy = (self.height - 50) // 2
            surface.blit(self._ready_overlay, (ox, oy))
            ready_rect = self._ready_txt.get_rect(center=(self.width // 2, self.height // 2))
            surface.blit(self._ready_txt, ready_rect)

        return surface

    def close(self):
        pygame.display.quit()
        pygame.quit()
