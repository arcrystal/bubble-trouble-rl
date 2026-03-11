"""Pygame-based renderer for the headless Bubble Trouble engine.

This module is only imported when render_mode is set — it is NOT needed for training.
"""

import numpy as np
import pygame
from config import (
    POWERUP_DOUBLE_HARPOON, POWERUP_FORCE_FIELD, POWERUP_HOURGLASS,
    DEFAULT_FPS,
)

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 200, 0)
BLUE = (100, 180, 255)
YELLOW = (245, 237, 7)
ORANGE = (237, 141, 45)
PURPLE = (160, 50, 220)
CYAN = (0, 220, 220)
GRAY = (120, 120, 120)

BALL_COLORS_BY_LEVEL = {
    1: YELLOW,
    2: BLUE,
    3: GREEN,
    4: RED,
}

POWERUP_COLORS = {
    POWERUP_DOUBLE_HARPOON: ORANGE,
    POWERUP_FORCE_FIELD: CYAN,
    POWERUP_HOURGLASS: PURPLE,
}

POWERUP_LABELS = {
    POWERUP_DOUBLE_HARPOON: "2x",
    POWERUP_FORCE_FIELD: "S",
    POWERUP_HOURGLASS: "T",
}


class PygameRenderer:
    """Renders engine state using pygame."""

    def __init__(self, width, height, fps=DEFAULT_FPS):
        self.width = width
        self.height = height
        self.fps = fps

        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Bubble Trouble RL")
        self.clock = pygame.time.Clock()

        self.font = pygame.font.Font(None, 24)
        self.big_font = pygame.font.Font(None, 36)

        # Load agent sprites if available, fall back to rectangle
        self.agent_sprites = {}
        try:
            for name in ("left", "still", "right"):
                img = pygame.image.load(f"Sprites/{name}.png")
                self.agent_sprites[name] = img
        except (pygame.error, FileNotFoundError):
            self.agent_sprites = None

    def render(self, state):
        """Render the game state to the display window."""
        surface = self._draw(state)
        self.window.blit(surface, (0, 0))
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.fps)

    def render_to_array(self, state):
        """Render to a numpy RGB array."""
        surface = self._draw(state)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2)
        )

    def _draw(self, state):
        """Draw all game elements to a surface."""
        surface = pygame.Surface((self.width, self.height))
        surface.fill(BLACK)

        # Draw lasers
        for i in range(len(state["laser_active"])):
            if state["laser_active"][i]:
                lx = state["laser_x"][i]
                ll = state["laser_length"][i]
                rect = pygame.Rect(lx, self.height - ll, 2, ll)
                pygame.draw.rect(surface, RED, rect)

        # Draw agent
        ax = state["agent_x"]
        aw = state["agent_w"]
        ah = state["agent_h"]
        agent_rect = pygame.Rect(ax, self.height - ah, aw, ah)

        if self.agent_sprites:
            sprite = self.agent_sprites["still"]
            scaled = pygame.transform.scale(sprite, (aw, ah))
            surface.blit(scaled, agent_rect)
        else:
            pygame.draw.rect(surface, WHITE, agent_rect)

        # Draw force field indicator
        if state.get("has_force_field"):
            pygame.draw.rect(surface, CYAN, agent_rect, 2)

        # Draw balls
        for i in range(state["n_balls"]):
            cx = int(state["ball_x"][i] + state["ball_radius"][i])
            cy = int(state["ball_y"][i] + state["ball_radius"][i])
            r = int(state["ball_radius"][i])
            lvl = state["ball_level"][i]
            color = BALL_COLORS_BY_LEVEL.get(lvl, WHITE)
            pygame.draw.circle(surface, color, (cx, cy), r)

        # Draw ground power-up
        if state.get("powerup_on_ground"):
            px = int(state["powerup_ground_x"])
            py = int(state["powerup_ground_y"])
            ptype = state["powerup_ground_type"]
            color = POWERUP_COLORS.get(ptype, WHITE)
            pygame.draw.circle(surface, color, (px, py), 8)
            label = POWERUP_LABELS.get(ptype, "?")
            txt = self.font.render(label, True, BLACK)
            txt_rect = txt.get_rect(center=(px, py))
            surface.blit(txt, txt_rect)

        # Draw HUD
        self._draw_hud(surface, state)

        return surface

    def _draw_hud(self, surface, state):
        """Draw heads-up display: level, timer, power-up indicators."""
        # Level indicator
        level_text = self.font.render(f"Level {state['current_level']}", True, WHITE)
        surface.blit(level_text, (5, 5))

        # Timer bar
        if state["max_steps"] > 0:
            remaining = max(0, state["max_steps"] - state["steps"])
            ratio = remaining / state["max_steps"]
            bar_width = self.width - 120
            bar_x = 80
            bar_y = 5
            bar_h = 10
            # Background
            pygame.draw.rect(surface, GRAY, (bar_x, bar_y, bar_width, bar_h))
            # Fill
            color = GREEN if ratio > 0.3 else RED
            pygame.draw.rect(surface, color, (bar_x, bar_y, int(bar_width * ratio), bar_h))

        # Active power-up indicators (top right)
        indicators = []
        if state.get("has_double_harpoon"):
            indicators.append(("2x", ORANGE))
        if state.get("has_force_field"):
            indicators.append(("Shield", CYAN))
        if state.get("hourglass_active"):
            t = state.get("hourglass_timer", 0)
            indicators.append((f"Slow {t:.1f}s", PURPLE))

        for i, (label, color) in enumerate(indicators):
            txt = self.font.render(label, True, color)
            surface.blit(txt, (self.width - 80, 20 + i * 20))

    def close(self):
        pygame.display.quit()
        pygame.quit()
