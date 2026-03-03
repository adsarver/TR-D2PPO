"""
weight_initializer.py — Behavioral Cloning Pre-training for DiffusionLSTM
==========================================================================
Collects expert demonstrations from the GapFollowPurePursuit baseline
across multiple maps, then trains the DiffusionLSTM diffusion actor using
the D²PPO Stage-1 objective:

    L = L_diff + λ · L_disp           (Eq. 6, Zou et al. 2025)

Usage:
    python weight_initializer.py                        # default settings
    python weight_initializer.py --epochs 200 --maps Hockenheim Monza
    python weight_initializer.py --load demos.pt        # skip collection, train from saved demos

Outputs:
    models/actor/pretrained/actor_pretrained.pt   — pretrained actor weights
    demos/expert_demos.pt                         — collected demonstrations (reusable)
"""

import argparse
import os
import time
import random
import math
import numpy as np
import torch
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.nn.functional as F
import gym

from models.AuxModels import VisionEncoder
from models.DiffusionLSTM import DiffusionLSTM
from baselines.gap_follow_pure_pursuit import GapFollowPurePursuit
from utils.utils import get_map_dir, generate_start_poses
from utils.diffusion_utils import extract

# Dispersive loss functions (same as D2PPO_agent.py)
def dispersive_loss_infonce_l2(features, temperature=0.5):
    B = features.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=features.device)
    diff = features.unsqueeze(0) - features.unsqueeze(1)
    sq_dist = (diff ** 2).sum(dim=-1)
    mask = ~torch.eye(B, dtype=torch.bool, device=features.device)
    sq_dist_masked = sq_dist[mask].reshape(B, B - 1)
    log_exp = -sq_dist_masked / temperature
    loss = torch.logsumexp(log_exp.reshape(-1), dim=0) - math.log(B * (B - 1))
    return loss


# ──────────────────────────────────────────────────────────────────────
# Vehicle / sim parameters  (must match train.py)
# ──────────────────────────────────────────────────────────────────────
PARAMS_DICT = {
    'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562,
    'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74,
    'I': 0.04712, 's_min': -0.34, 's_max': 0.34,
    'sv_min': -3.2, 'sv_max': 3.2, 'v_switch': 7.319,
    'a_max': 9.51, 'v_min': -5.0, 'v_max': 20.0,
    'width': 0.31, 'length': 0.58,
}

NUM_BEAMS  = 1080
LIDAR_FOV  = 4.7
STATE_DIM  = 4


# ──────────────────────────────────────────────────────────────────────
# Demo collection  (multiprocess — one env per map, runs in parallel)
# ──────────────────────────────────────────────────────────────────────

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


def _collect_single_map(
    map_name: str,
    num_agents: int,
    steps_per_map: int,
    collision_reset_threshold: int,
) -> list[dict]:
    """
    Worker function executed in its own process.
    Creates an isolated gym env + expert controller for *map_name* and
    collects (scan, state, action) demonstrations.
    """
    # Each process must import gym afresh (env has C state)
    import gym as _gym
    demos: list[dict] = []

    env = _gym.make(
        "f110_gym:f110-v0",
        map=get_map_dir(map_name) + f"/{map_name}_map",
        num_agents=num_agents,
        num_beams=NUM_BEAMS,
        fov=LIDAR_FOV,
        params=PARAMS_DICT,
    )

    expert = GapFollowPurePursuit(
        map_name=map_name,
        wheelbase=PARAMS_DICT['lf'] + PARAMS_DICT['lr'],
        max_steering=PARAMS_DICT['s_max'],
        max_speed=8.0,
        min_speed=1.5,
        num_beams=NUM_BEAMS,
        fov=LIDAR_FOV,
    )

    poses = generate_start_poses(map_name, num_agents)
    obs, _, _, _ = env.reset(poses=poses)

    collision_timers = np.zeros(num_agents, dtype=np.int32)

    for step in range(steps_per_map):
        actions = expert.get_actions_batch(obs)  # (N, 2)

        for a_idx in range(num_agents):
            scan_np = np.asarray(obs["scans"][a_idx], dtype=np.float32)
            state_np = np.array([
                obs["linear_vels_x"][a_idx],
                obs["linear_vels_y"][a_idx],
                obs["ang_vels_z"][a_idx],
                obs["linear_accel_x"][a_idx],
            ], dtype=np.float32)
            action_np = actions[a_idx].astype(np.float32)

            demos.append({
                "scan":   scan_np,
                "state":  state_np,
                "action": action_np,
                "map":    map_name,
            })

        next_obs, _, _, _ = env.step(actions)

        # Collision / stuck detection → reset
        collisions = np.array(next_obs["collisions"][:num_agents])
        velocities = np.array(next_obs["linear_vels_x"][:num_agents])
        stuck = (collisions == 1) | (np.abs(velocities) < 0.1)
        collision_timers[stuck] += 1
        collision_timers[~stuck] = 0

        to_reset = np.where(collision_timers >= collision_reset_threshold)[0]
        if len(to_reset) > 0:
            cur_poses = np.stack([
                next_obs["poses_x"], next_obs["poses_y"], next_obs["poses_theta"]
            ], axis=1)
            new_poses = generate_start_poses(map_name, num_agents, agent_poses=cur_poses)
            next_obs, _, _, _ = env.reset(poses=new_poses, agent_idxs=to_reset)
            collision_timers[to_reset] = 0

        obs = next_obs

    env.close()
    return demos


def collect_demos(
    maps: list[str],
    num_agents: int = 4,
    steps_per_map: int = 2000,
    collision_reset_threshold: int = 32,
    max_workers: int | None = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Collect demonstrations from GapFollowPurePursuit across all *maps*
    using one **process per map** for full parallelism (each process owns
    its own gym environment, avoiding any shared-state issues).

    Args:
        max_workers: Number of parallel processes.
            ``None`` → ``min(len(maps), cpu_count())``.
    """
    if max_workers is None:
        max_workers = min(len(maps), mp.cpu_count())
    max_workers = max(1, max_workers)

    if verbose:
        print(f"\n[Demo] Collecting on {len(maps)} maps with {max_workers} parallel workers "
              f"({num_agents} agents × {steps_per_map} steps each)")

    demos: list[dict] = []

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _collect_single_map,
                map_name, num_agents, steps_per_map, collision_reset_threshold,
            ): map_name
            for map_name in maps
        }

        for future in as_completed(futures):
            map_name = futures[future]
            try:
                map_demos = future.result()
                demos.extend(map_demos)
                if verbose:
                    print(f"  ✓ {map_name}: {len(map_demos)} demos  (total so far: {len(demos)})")
            except Exception as exc:
                print(f"  ✗ {map_name} failed: {exc}")

    if verbose:
        print(f"\n[Demo] Total demonstrations collected: {len(demos)}")
    return demos


# ──────────────────────────────────────────────────────────────────────
# Training loop  (D²PPO Stage 1 — behavioural cloning)
# ──────────────────────────────────────────────────────────────────────

def pretrain(
    demos: list[dict],
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 3e-4,
    dispersive_lambda: float = 0.5,
    dispersive_temperature: float = 0.5,
    num_diffusion_steps: int = 25,
    gradient_accumulation_steps: int = 1,
    save_dir: str = "models/actor/pretrained",
    save_name: str = "actor_pretrained.pt",
    device: str | torch.device = "cuda",
):
    """
    Train a fresh DiffusionLSTM on the collected demonstrations.

    The loss is the standard DDPM noise-prediction MSE (L_diff) plus
    dispersive regularisation on intermediate denoise-MLP features (L_disp).
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"\n[Pretrain] Device: {device}")
    print(f"[Pretrain] {len(demos)} demos, {epochs} epochs, batch_size={batch_size}, lr={lr}")

    # ── Build model ──────────────────────────────────────────────────
    model = DiffusionLSTM(
            state_dim=STATE_DIM,
            action_dim=2,
            num_diffusion_steps=num_diffusion_steps,
            time_emb_dim=32,
            hidden_dims=(512, 512, 512),
            beta_schedule="cosine",
            odom_expand=64,
            lstm_hidden_size=256,
            lstm_num_layers=2,
            memory_length=350,
            memory_stride=20
        ).to(device)

    # Register dispersive hooks on the denoising MLP (last block)
    model.denoise_net.register_dispersive_hooks("late")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ── Materialise tensors ──────────────────────────────────────────
    all_scans   = torch.stack([torch.from_numpy(d["scan"])   for d in demos]).unsqueeze(1)  # (N, 1, 1080)
    all_states  = torch.stack([torch.from_numpy(d["state"])  for d in demos])               # (N, 4)
    all_actions = torch.stack([torch.from_numpy(d["action"]) for d in demos])               # (N, 2)
    N = len(demos)
    print(f"[Pretrain] Tensors ready — scans {tuple(all_scans.shape)}, "
          f"states {tuple(all_states.shape)}, actions {tuple(all_actions.shape)}")

    # ── Training ─────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float("inf")
    torch.backends.cudnn.benchmark = True

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(N)
        epoch_diff = 0.0
        epoch_disp = 0.0
        num_batches = 0
        optimizer.zero_grad(set_to_none=True)

        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            scan_b   = all_scans[idx].to(device, non_blocking=True)
            state_b  = all_states[idx].to(device, non_blocking=True)
            action_b = all_actions[idx].to(device, non_blocking=True)
            B = scan_b.shape[0]

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                # Encode observations (no temporal continuity for shuffled batches)
                obs_features, _, _, _ = model.encode_observation(
                    scan_b, state_b, obs_buffer=None, hidden_h=None, hidden_c=None,
                )

                # ── Diffusion loss (noise prediction MSE) ───────────
                # compute_diffusion_loss normalises actions internally
                diff_loss = model.compute_diffusion_loss(action_b, obs_features)

                # ── Dispersive loss on intermediate features ────────
                action_norm = model.normalize_action(action_b)
                t_rand = torch.randint(0, num_diffusion_steps, (B,), device=device)
                noise  = torch.randn_like(action_norm)
                noisy  = model.q_sample(action_norm, t_rand, noise=noise)
                _      = model.denoise_net(noisy, obs_features, t_rand)

                feat_dict = model.denoise_net.get_intermediate_features()
                if isinstance(feat_dict, list):
                    # get_intermediate_features may return a list; convert to dict
                    feat_dict = {i: f for i, f in enumerate(feat_dict)}
                disp_loss = torch.tensor(0.0, device=device)
                for _, feats in feat_dict.items():
                    if feats.ndim > 2:
                        feats = feats.mean(dim=list(range(1, feats.ndim - 1)))
                    disp_loss = disp_loss + dispersive_loss_infonce_l2(feats, dispersive_temperature)
                if len(feat_dict) > 0:
                    disp_loss = disp_loss / len(feat_dict)

                loss = (diff_loss + dispersive_lambda * disp_loss) / gradient_accumulation_steps

            scaler.scale(loss).backward()

            # Gradient accumulation step
            if (num_batches + 1) % gradient_accumulation_steps == 0 or (start + batch_size >= N):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            epoch_diff += diff_loss.item() * B
            epoch_disp += disp_loss.item() * B
            num_batches += 1

        scheduler.step()

        avg_diff = epoch_diff / N
        avg_disp = epoch_disp / N
        avg_total = avg_diff + dispersive_lambda * avg_disp

        # Checkpoint best
        if avg_total < best_loss:
            best_loss = avg_total
            torch.save(model.state_dict(), os.path.join(save_dir, save_name))
            marker = " ★"
        else:
            marker = ""

        if epoch % max(1, epochs // 20) == 0 or epoch <= 5 or marker:
            print(
                f"  Epoch {epoch:>4d}/{epochs}  "
                f"diff={avg_diff:.5f}  disp={avg_disp:.5f}  "
                f"total={avg_total:.5f}  lr={scheduler.get_last_lr()[0]:.2e}{marker}"
            )

        # Periodic checkpoint every 25 epochs
        if epoch % 25 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"actor_epoch_{epoch}.pt"))

    # Final save
    torch.save(model.state_dict(), os.path.join(save_dir, "actor_final.pt"))
    print(f"\n[Pretrain] Done.  Best total loss: {best_loss:.5f}")
    print(f"  Saved to {os.path.join(save_dir, save_name)}")
    return model


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    ALL_MAPS = [
        "Hockenheim", "Monza", "Melbourne", "BrandsHatch",
        "Oschersleben", "Sakhir", "Sepang", "SaoPaulo",
        "Budapest", "Catalunya", "Silverstone",
        "Zandvoort", "MoscowRaceway", "Austin", "Nuerburgring",
        "Spa", "YasMarina", "Sochi",
    ]

    parser = argparse.ArgumentParser(description="D²PPO Stage 1: BC Pre-training from GapFollowPurePursuit")
    # Demo collection
    parser.add_argument("--maps", nargs="+", default=ALL_MAPS,
                        help="Maps to collect demos on (default: all)")
    parser.add_argument("--num_agents", type=int, default=4,
                        help="Agents per env during collection")
    parser.add_argument("--steps_per_map", type=int, default=8000,
                        help="Env steps per map during collection")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Parallel processes for demo collection (default: min(num_maps, cpu_count))")
    parser.add_argument("--load", type=str, default=None,
                        help="Path to a saved demos .pt file (skip collection)")
    parser.add_argument("--save_demos", type=str, default="demos/expert_demos.pt",
                        help="Where to save collected demos for reuse")
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dispersive_lambda", type=float, default=0.1)
    parser.add_argument("--num_diffusion_steps", type=int, default=25)
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    # Output
    parser.add_argument("--save_dir", type=str, default="models/actor/pretrained")
    parser.add_argument("--save_name", type=str, default="actor_pretrained.pt")

    args = parser.parse_args()

    # ── Collect or load demonstrations ───────────────────────────────
    if args.load and os.path.exists(args.load):
        print(f"[Main] Loading demos from {args.load}")
        demos = torch.load(args.load, weights_only=False)
        print(f"[Main] Loaded {len(demos)} demos")
    else:
        demos = collect_demos(
            maps=args.maps,
            num_agents=args.num_agents,
            steps_per_map=args.steps_per_map,
            max_workers=args.max_workers,
        )
        # Save demos for reuse
        os.makedirs(os.path.dirname(args.save_demos) or ".", exist_ok=True)
        torch.save(demos, args.save_demos)
        print(f"[Main] Demos saved to {args.save_demos}")

    # ── Train ────────────────────────────────────────────────────────
    model = pretrain(
        demos=demos,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dispersive_lambda=args.dispersive_lambda,
        num_diffusion_steps=args.num_diffusion_steps,
        gradient_accumulation_steps=args.gradient_accumulation,
        save_dir=args.save_dir,
        save_name=args.save_name,
    )

    # ── Quick validation: sample actions from the trained model ──────
    print("\n[Validation] Sampling actions from pretrained model …")
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        # Pick 8 random demos
        sample_idx = random.sample(range(len(demos)), min(8, len(demos)))
        scans  = torch.stack([torch.from_numpy(demos[i]["scan"]) for i in sample_idx]).unsqueeze(1).to(device)
        states = torch.stack([torch.from_numpy(demos[i]["state"]) for i in sample_idx]).to(device)
        expert = torch.stack([torch.from_numpy(demos[i]["action"]) for i in sample_idx]).to(device)

        obs_feat, _, _, _ = model.encode_observation(scans, states, obs_buffer=None)
        sampled = model.sample_action(obs_feat, deterministic=True)

        print(f"  Expert actions (first 4): {expert[:4].cpu().numpy()}")
        print(f"  Model actions  (first 4): {sampled[:4].cpu().numpy()}")
        mse = F.mse_loss(sampled, expert).item()
        print(f"  Action MSE: {mse:.5f}")

    print("\n✓ Pre-training complete. Use the saved weights with D2PPO_agent via:")
    print(f'    transfer=["{os.path.join(args.save_dir, args.save_name)}", None]')


if __name__ == "__main__":
    main()
