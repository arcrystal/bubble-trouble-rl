"""DeepSets feature extractor for Bubble Trouble RL.

Three-path architecture:
  1. Ball DeepSets: per-ball MLP → sum+max+mean pooling → 3×hidden ball summary
  2. Obstacle DeepSets: per-obstacle MLP → sum+max pooling → 2×hidden obstacle summary
  3. Scalar context MLP: agent+global+powerup → 2-layer MLP → context output

Ball and obstacle paths are permutation-invariant (DeepSets). Inactive
slots are masked before pooling. Mean pooling captures "average threat level"
that sum/max alone cannot represent.
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

from config import (
    MAX_OBS_BALLS, OBS_PER_BALL, OBS_PER_OBSTACLE,
    OBS_AGENT, OBS_GLOBAL, OBS_POWERUP, MAX_OBSTACLES,
)


# Index of the is_active flag within each ball's feature vector
_IS_ACTIVE_IDX = 6

# Index of the width feature within each obstacle's feature vector (used for activity detection)
_OBS_WIDTH_IDX = 2


class BubbleFeatureExtractor(BaseFeaturesExtractor):
    """Three-path DeepSets feature extractor.

    Splits the flat observation into ball features, obstacle features, and
    scalar context. Balls and obstacles are processed through separate shared
    MLPs with permutation-invariant pooling. Scalar context goes through a
    2-layer MLP.

    Output: concat(ball_summary, obstacle_summary, context_output)
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        per_ball_hidden: int = 64,
        per_obstacle_hidden: int = 32,
        context_hidden: int = 64,
        context_output: int = 32,
    ):
        # Output: 3×ball_hidden (sum+max+mean) + 2×obstacle_hidden (sum+max) + context_output
        features_dim = 3 * per_ball_hidden + 2 * per_obstacle_hidden + context_output
        super().__init__(observation_space, features_dim)

        self.n_balls = MAX_OBS_BALLS
        self.per_ball = OBS_PER_BALL
        self.n_obstacles = MAX_OBSTACLES
        self.per_obstacle = OBS_PER_OBSTACLE
        self.scalar_context_dim = OBS_AGENT + OBS_GLOBAL + OBS_POWERUP

        # Shared per-ball MLP (applied identically to each ball slot)
        self.ball_mlp = nn.Sequential(
            nn.Linear(OBS_PER_BALL, per_ball_hidden),
            nn.ReLU(),
            nn.Linear(per_ball_hidden, per_ball_hidden),
            nn.ReLU(),
        )

        # Shared per-obstacle MLP (applied identically to each obstacle slot)
        self.obstacle_mlp = nn.Sequential(
            nn.Linear(OBS_PER_OBSTACLE, per_obstacle_hidden),
            nn.ReLU(),
            nn.Linear(per_obstacle_hidden, per_obstacle_hidden),
            nn.ReLU(),
        )

        # 2-layer context MLP for scalar features (agent + global + powerup)
        self.context_mlp = nn.Sequential(
            nn.Linear(self.scalar_context_dim, context_hidden),
            nn.ReLU(),
            nn.Linear(context_hidden, context_output),
            nn.ReLU(),
        )

        self.per_ball_hidden = per_ball_hidden
        self.per_obstacle_hidden = per_obstacle_hidden

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]

        # --- Split observation into three sections ---
        ball_end = self.n_balls * self.per_ball
        obs_start = ball_end + self.scalar_context_dim

        ball_flat = observations[:, :ball_end]                        # (B, 896)
        scalar_context = observations[:, ball_end:obs_start]          # (B, 24)
        obstacle_flat = observations[:, obs_start:]                   # (B, 48)

        # --- Ball path: DeepSets with sum+max+mean pooling ---
        # Reshape: feature-major → (B, per_ball, n_balls) → transpose → (B, n_balls, per_ball)
        ball_features = ball_flat.view(batch_size, self.per_ball, self.n_balls).transpose(1, 2)

        # Extract is_active mask: (B, n_balls, 1)
        is_active = ball_features[:, :, _IS_ACTIVE_IDX].unsqueeze(-1)

        # Per-ball MLP: (B, n_balls, per_ball) → (B, n_balls, hidden)
        ball_embeddings = self.ball_mlp(ball_features)
        ball_embeddings = ball_embeddings * is_active  # mask inactive

        # Sum pooling
        ball_sum = ball_embeddings.sum(dim=1)                         # (B, hidden)

        # Max pooling (inactive → large negative so they don't win)
        masked_for_max = ball_embeddings + (1 - is_active) * (-1e6)
        ball_max, _ = masked_for_max.max(dim=1)                       # (B, hidden)
        any_active = is_active.any(dim=1)                             # (B, 1)
        ball_max = ball_max * any_active.float()

        # Mean pooling (sum / active_count, guarded for div-by-zero)
        active_count = is_active.sum(dim=1).clamp(min=1.0)           # (B, 1)
        ball_mean = ball_sum / active_count                           # (B, hidden)

        ball_summary = torch.cat([ball_sum, ball_max, ball_mean], dim=-1)  # (B, 3*hidden)

        # --- Obstacle path: DeepSets with sum+max pooling ---
        # Reshape: (B, 48) → (B, 8, 6)
        obs_features = obstacle_flat.view(batch_size, self.n_obstacles, self.per_obstacle)

        # Detect active obstacles by width > threshold (zero-filled slots have width = -1.0 after normalization)
        obs_is_active = (obs_features[:, :, _OBS_WIDTH_IDX] > -0.99).unsqueeze(-1)  # (B, 8, 1)

        # Per-obstacle MLP
        obs_embeddings = self.obstacle_mlp(obs_features)
        obs_embeddings = obs_embeddings * obs_is_active.float()

        obs_sum = obs_embeddings.sum(dim=1)                           # (B, obs_hidden)
        masked_obs_max = obs_embeddings + (1 - obs_is_active.float()) * (-1e6)
        obs_max, _ = masked_obs_max.max(dim=1)                       # (B, obs_hidden)
        any_obs_active = obs_is_active.any(dim=1)                     # (B, 1)
        obs_max = obs_max * any_obs_active.float()

        obs_summary = torch.cat([obs_sum, obs_max], dim=-1)           # (B, 2*obs_hidden)

        # --- Scalar context path: 2-layer MLP ---
        context_out = self.context_mlp(scalar_context)                # (B, context_output)

        return torch.cat([ball_summary, obs_summary, context_out], dim=-1)  # (B, features_dim)
