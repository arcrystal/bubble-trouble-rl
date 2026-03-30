"""Behavioral Cloning pretraining from human demonstrations.

Trains the MaskablePPO policy architecture via supervised learning on
recorded (observation, action) pairs. The pretrained weights serve as
a warm start for PPO fine-tuning.

Two training phases:

  Phase 1 — Actor (features_extractor + policy_net + action_net):
    - Weighted cross-entropy: inverse-frequency shoot class weight + 3x multiplier
    - Cosine annealing LR schedule with gradient clipping (max_norm=1.0)
    - Observation noise augmentation (std=0.02) for generalization
    - Label smoothing (0.05 for both move and shoot — 0.1 was too aggressive for rare shoot timing)
    - AdamW with weight decay 1e-4
    - 90/10 train/val split with early stopping (patience 15)

  Phase 2 — Value function (value_net, frozen features_extractor):
    - Monte Carlo returns from demo rewards (requires rewards in .npz)
    - Normalized to approximate VecNormalize scale
    - 30 epochs MSE regression — approximate is fine, PPO refines it

The pretrained model is then used via:
  python src/train_ppo.py --warmup bc_pretrained.zip --demo demo_*.npz

The --demo flag enables DAPG (Demo Augmented Policy Gradient): a decaying
BC auxiliary loss runs alongside PPO to prevent catastrophic forgetting of
demo-learned skills during early fine-tuning.

Usage:
  python src/pretrain_bc.py checkpoints/user_warmups/demo_*.npz
  python src/pretrain_bc.py checkpoints/user_warmups/demo_*.npz --max-epochs 200 --lr 1e-3
  python src/pretrain_bc.py demos/*.npz --output checkpoints/user_warmups/bc_pretrained
"""

import argparse
import copy
import os
import glob

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Subset

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from bubble_env import BubbleTroubleEnv
from config import TRAINING as T
from feature_extractor import BubbleFeatureExtractor


def load_demos(paths):
    """Load and concatenate demonstrations from one or more .npz files."""
    all_obs, all_act, all_rew, all_lvl = [], [], [], []
    has_rewards = True
    for path in paths:
        data = np.load(path)
        all_obs.append(data["observations"])
        all_act.append(data["actions"])
        levels = np.unique(data.get("level_ids", []))
        if "rewards" in data:
            all_rew.append(data["rewards"])
        else:
            has_rewards = False
        if "level_ids" in data:
            all_lvl.append(data["level_ids"])
        print(f"  {path}: {len(data['actions']):,} steps, levels {levels.tolist()}"
              f"{'  (has rewards)' if 'rewards' in data else ''}")
    obs = np.concatenate(all_obs, axis=0).astype(np.float32)
    act = np.concatenate(all_act, axis=0).astype(np.int64)
    rewards = np.concatenate(all_rew, axis=0).astype(np.float32) if has_rewards else None
    level_ids = np.concatenate(all_lvl, axis=0).astype(np.int32) if all_lvl else None
    if not has_rewards:
        print("  [NOTE] Some files lack rewards — value pretraining will be skipped")
    print(f"  Total: {len(act):,} demonstration steps")
    return obs, act, rewards, level_ids


def compute_mc_returns(rewards, level_ids, gamma=0.999):
    """Compute discounted Monte Carlo returns, split by level boundaries."""
    returns = np.zeros_like(rewards)
    if level_ids is not None:
        boundaries = np.where(np.diff(level_ids) != 0)[0] + 1
        boundaries = np.concatenate([[0], boundaries, [len(rewards)]])
    else:
        boundaries = np.array([0, len(rewards)])

    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        G = 0.0
        for t in range(end - 1, start - 1, -1):
            G = rewards[t] + gamma * G
            returns[t] = G
    return returns


def _mask_fn(env):
    return env.action_masks()


def build_model():
    """Create a MaskablePPO with the training architecture (for weight init)."""
    def _make():
        env = BubbleTroubleEnv()
        return ActionMasker(env, _mask_fn)
    venv = DummyVecEnv([_make])
    venv = VecNormalize(venv, norm_obs=False, norm_reward=False)
    model = MaskablePPO(
        policy="MlpPolicy",
        env=venv,
        policy_kwargs=dict(
            features_extractor_class=BubbleFeatureExtractor,
            features_extractor_kwargs=dict(
                per_ball_hidden=T["per_ball_hidden"],
                per_obstacle_hidden=T["per_obstacle_hidden"],
                context_hidden=T["context_hidden"],
                context_output=T["context_output"],
            ),
            net_arch=dict(pi=T["net_arch_pi"], vf=T["net_arch_vf"]),
            activation_fn=torch.nn.ReLU,
        ),
        verbose=0,
        device="cpu",
    )
    venv.close()
    return model


def train_bc(model, obs, actions, max_epochs, lr, batch_size, device,
             obs_noise_std=0.02, patience=15):
    """Train the policy network via weighted cross-entropy on demo data."""
    n = len(obs)
    indices = np.random.permutation(n)
    n_val = max(n // 10, 1)
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    obs_t = torch.tensor(obs, device=device)
    move_t = torch.tensor(actions[:, 0], device=device)
    shoot_t = torch.tensor(actions[:, 1], device=device)

    dataset = TensorDataset(obs_t, move_t, shoot_t)
    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(
        Subset(dataset, val_idx.tolist()),
        batch_size=batch_size, shuffle=False)

    # Compute shoot class weights from training data
    train_shoot = actions[train_idx, 1]
    shoot_counts = np.bincount(train_shoot, minlength=2)
    SHOOT_MULTIPLIER = 3.0
    shoot_weight = torch.tensor(
        [shoot_counts[1] / max(shoot_counts[0], 1) * SHOOT_MULTIPLIER, 1.0],
        dtype=torch.float32, device=device)
    print(f"  Shoot class weight: {shoot_weight[0].item():.1f}x "
          f"(shoot={shoot_counts[0]:,}, no_shoot={shoot_counts[1]:,}, "
          f"multiplier={SHOOT_MULTIPLIER}x)")
    print(f"  Train/val split: {len(train_idx):,} / {len(val_idx):,}")

    policy = model.policy.to(device)

    params = (
        list(policy.features_extractor.parameters()) +
        list(policy.mlp_extractor.policy_net.parameters()) +
        list(policy.action_net.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    # Fix E: Linear warmup over first 5 epochs then cosine decay, avoiding
    # large gradient spikes from the heavily-weighted shoot loss at init.
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        cos_progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return (1 + np.cos(np.pi * cos_progress)) / 2 * (1 - 1/20) + 1/20
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    policy.train()
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        # --- Training ---
        total_loss = 0.0
        move_correct = 0
        shoot_correct = 0
        shoot_tp = 0
        shoot_total = 0
        n_samples = 0

        for obs_b, move_b, shoot_b in train_loader:
            # Observation noise augmentation
            obs_noisy = obs_b + torch.randn_like(obs_b) * obs_noise_std

            features = policy.extract_features(obs_noisy)
            latent_pi = policy.mlp_extractor.forward_actor(features)
            logits = policy.action_net(latent_pi)

            move_logits = logits[:, :3]
            shoot_logits = logits[:, 3:]

            loss = (F.cross_entropy(move_logits, move_b, label_smoothing=0.05) +
                    F.cross_entropy(shoot_logits, shoot_b,
                                    weight=shoot_weight, label_smoothing=0.05))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            bs = len(obs_b)
            total_loss += loss.item() * bs
            move_correct += (move_logits.argmax(1) == move_b).sum().item()
            shoot_correct += (shoot_logits.argmax(1) == shoot_b).sum().item()
            shoot_mask = (shoot_b == 0)
            shoot_tp += (shoot_logits.argmax(1)[shoot_mask] == 0).sum().item()
            shoot_total += shoot_mask.sum().item()
            n_samples += bs

        scheduler.step()

        train_loss = total_loss / max(n_samples, 1)
        move_acc = move_correct / max(n_samples, 1) * 100
        shoot_acc = shoot_correct / max(n_samples, 1) * 100
        shoot_recall = shoot_tp / max(shoot_total, 1) * 100

        # --- Validation ---
        policy.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for obs_b, move_b, shoot_b in val_loader:
                features = policy.extract_features(obs_b)
                latent_pi = policy.mlp_extractor.forward_actor(features)
                logits = policy.action_net(latent_pi)
                move_logits = logits[:, :3]
                shoot_logits = logits[:, 3:]
                loss = (F.cross_entropy(move_logits, move_b) +
                        F.cross_entropy(shoot_logits, shoot_b, weight=shoot_weight))
                val_loss_sum += loss.item() * len(obs_b)
                val_n += len(obs_b)
        val_loss = val_loss_sum / max(val_n, 1)
        policy.train()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.policy.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        cur_lr = optimizer.param_groups[0]["lr"]
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == max_epochs - 1 or patience_counter == 0:
            print(f"  Epoch {epoch+1:3d}/{max_epochs}: "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"move={move_acc:.1f}%  shoot={shoot_acc:.1f}%  "
                  f"shoot_recall={shoot_recall:.1f}%  lr={cur_lr:.1e}"
                  f"{'  *best*' if patience_counter == 0 else ''}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    # Restore best weights
    if best_state is not None:
        model.policy.load_state_dict(best_state)
    policy.eval()
    return best_val_loss


def train_value(model, obs, returns, batch_size, device, epochs=30):
    """Train value function via MSE on Monte Carlo returns (frozen extractor)."""
    # Normalize returns
    ret_mean, ret_std = returns.mean(), returns.std()
    returns_norm = (returns - ret_mean) / max(ret_std, 1e-8)

    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    ret_t = torch.tensor(returns_norm, dtype=torch.float32, device=device)

    dataset = TensorDataset(obs_t, ret_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    policy = model.policy.to(device)

    # Freeze features extractor — tuned for actor, don't corrupt
    for p in policy.features_extractor.parameters():
        p.requires_grad = False

    value_params = (
        list(policy.mlp_extractor.value_net.parameters()) +
        list(policy.value_net.parameters())
    )
    optimizer = torch.optim.Adam(value_params, lr=1e-3)

    policy.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_samples = 0
        for obs_b, ret_b in loader:
            features = policy.extract_features(obs_b)
            latent_vf = policy.mlp_extractor.forward_critic(features)
            v_pred = policy.value_net(latent_vf).squeeze(-1)
            loss = F.mse_loss(v_pred, ret_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(obs_b)
            n_samples += len(obs_b)

        avg_loss = total_loss / max(n_samples, 1)
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1:3d}/{epochs}: value_mse={avg_loss:.4f}")

    # Unfreeze features extractor for PPO fine-tuning
    for p in policy.features_extractor.parameters():
        p.requires_grad = True

    policy.eval()
    print(f"  Return stats: mean={ret_mean:.2f}, std={ret_std:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain MaskablePPO policy via behavioral cloning"
    )
    parser.add_argument("demos", nargs="+",
                        help="Path(s) to demo .npz files (supports globs)")
    parser.add_argument("--max-epochs", type=int, default=200,
                        help="Max training epochs (default: 200, early stopping usually fires earlier)")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate (default: 5e-4)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Mini-batch size (default: 256)")
    parser.add_argument("--output", type=str,
                        default="checkpoints/user_warmups/bc_pretrained",
                        help="Output model path without .zip (default: checkpoints/user_warmups/bc_pretrained)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--no-value", action="store_true",
                        help="Skip value function pretraining even if rewards are available")
    args = parser.parse_args()

    # Resolve glob patterns (shell might not expand them)
    paths = []
    for p in args.demos:
        expanded = sorted(glob.glob(p))
        paths.extend(expanded if expanded else [p])

    print("Loading demonstrations...")
    obs, actions, rewards, level_ids = load_demos(paths)

    print("\nBuilding model (same architecture as training)...")
    model = build_model()
    n_params = sum(p.numel() for p in model.policy.parameters())
    actor_params = (
        sum(p.numel() for p in model.policy.features_extractor.parameters()) +
        sum(p.numel() for p in model.policy.mlp_extractor.policy_net.parameters()) +
        sum(p.numel() for p in model.policy.action_net.parameters())
    )
    value_params = (
        sum(p.numel() for p in model.policy.mlp_extractor.value_net.parameters()) +
        sum(p.numel() for p in model.policy.value_net.parameters())
    )
    print(f"  Total params: {n_params:,}  (actor: {actor_params:,}, value: {value_params:,})")

    # Phase 1: Actor BC training
    print(f"\nPhase 1: Actor BC Training ({args.max_epochs} max epochs, lr={args.lr}, "
          f"batch={args.batch_size}, device={args.device})")
    print(f"  Regularization: AdamW(wd=1e-4), obs_noise=0.02, "
          f"label_smooth=0.05 (both heads), warmup=5ep, early_stop(patience=15)")
    best = train_bc(model, obs, actions, args.max_epochs, args.lr,
                    args.batch_size, args.device)

    # Phase 2: Value function pretraining
    if rewards is not None and not args.no_value:
        print(f"\nPhase 2: Value Function Pretraining (30 epochs, MC returns, frozen extractor)")
        returns = compute_mc_returns(rewards, level_ids, gamma=0.999)
        train_value(model, obs, returns, args.batch_size, args.device, epochs=30)
    elif rewards is None:
        print("\nPhase 2: Skipped (no reward data in demos)")
    else:
        print("\nPhase 2: Skipped (--no-value flag)")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    model.save(args.output)
    print(f"\nSaved BC-pretrained model -> {args.output}.zip  (best val loss: {best:.4f})")


if __name__ == "__main__":
    main()
