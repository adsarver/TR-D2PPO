"""
weight_initializer.py — Behavioral Cloning Pre-training for DiffusionLSTM
==========================================================================
Collects expert demonstrations from the MPC baseline
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
from models.DiffusionMamba2 import DiffusionMamba2
from utils.diffusion_utils import cosine_beta_schedule, linear_beta_schedule
from utils.diffusion_utils import extract
from baselines.mpc_agent import MPCAgent
from utils.utils import get_map_dir, generate_start_poses

# Dispersive loss functions (same as D2PPO_agent.py)
def dispersive_loss_infonce_l2(features, temperature=0.5):
    B = features.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=features.device)
    # Project to unit sphere (standard in contrastive learning)
    features = F.normalize(features, dim=-1)
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
    num_noise_agents: int = 16,
) -> list[dict]:
    """
    Worker function executed in its own process.
    Creates an isolated gym env + expert controller for *map_name* and
    collects (scan, state, action) demonstrations.
    """
    # Each process must import gym afresh (env has C state)
    import gym as _gym
    demos: list[dict] = []
    
    total_agents = num_agents + num_noise_agents

    env = _gym.make(
        "f110_gym:f110-v0",
        map=get_map_dir(map_name) + f"/{map_name}_map",
        num_agents=total_agents,
        num_beams=NUM_BEAMS,
        fov=LIDAR_FOV,
        params=PARAMS_DICT,
    )

    expert = MPCAgent(
        map_name=map_name,
        wheelbase=PARAMS_DICT['lf'] + PARAMS_DICT['lr'],
        max_steering=PARAMS_DICT['s_max'],
        max_accel=PARAMS_DICT['a_max'],
        LIDAR_FOV=LIDAR_FOV,
        num_beams=NUM_BEAMS,
        horizon=8,
        speed_scale=0.8,
        emergency_dist=0.8,
    )
    
    noise_agents = MPCAgent(
        map_name=map_name,
        wheelbase=PARAMS_DICT['lf'] + PARAMS_DICT['lr'],
        max_steering=PARAMS_DICT['s_max'],
        max_accel=PARAMS_DICT['a_max'],
        LIDAR_FOV=LIDAR_FOV,
        num_beams=NUM_BEAMS,
        horizon=8,
        speed_scale=0.8,
        emergency_dist=0.8,
        speed_clamp=7.5
    )

    poses = generate_start_poses(map_name, total_agents)
    obs, _, _, _ = env.reset(poses=poses)

    collision_timers = np.zeros(total_agents, dtype=np.int32)

    for step in range(steps_per_map):
        actions = expert.get_actions_batch(obs)  # (N, 2)
        noise_actions = noise_agents.get_actions_batch(obs)
        actions[num_agents:] = noise_actions[:num_noise_agents]  # Add noise agents at the end
        # Last 3 noise agents act as static obstacles (zero steering & velocity)
        actions[-3:] = np.array([0.0, 0.0])
        
        next_obs, _, _, _ = env.step(actions)

        # Collision / stuck detection
        collisions = np.array(next_obs["collisions"])
        velocities = np.array(next_obs["linear_vels_x"])

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
                # Reward-relevant fields (for critic pretraining)
                "step":   step,
                "agent":  a_idx,
                "next_velocity":  float(velocities[:num_agents][a_idx]),
                "next_collision": int(collisions[:num_agents][a_idx]),
            })

        # Stuck detection → reset
        stuck = (collisions == 1) | (np.abs(velocities) < 0.1)
        collision_timers[stuck] += 1
        collision_timers[~stuck] = 0

        to_reset = np.where(collision_timers >= collision_reset_threshold)[0]
        if len(to_reset) > 0:
            cur_poses = np.stack([
                next_obs["poses_x"], next_obs["poses_y"], next_obs["poses_theta"]
            ], axis=1)
            new_poses = generate_start_poses(map_name, total_agents, agent_poses=cur_poses)
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
    Collect demonstrations from MPC across all *maps*
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
    model_type: str = "lstm",
    epochs: int = 100,
    tbtt_length: int = 64,
    checkpoint_every: int = 0,
    traj_batch_size: int = 8,
    lr: float = 3e-4,
    dispersive_lambda: float = 0.5,
    dispersive_temperature: float = 0.5,
    num_diffusion_steps: int = 10,
    gradient_accumulation_steps: int = 1,
    save_dir: str = "models/actor/pretrained",
    save_name: str = "actor_pretrained.pt",
    device: str | torch.device = "cuda",
):
    """
    Train a fresh DiffusionLSTM or DiffusionMamba2 on the collected demonstrations
    using Truncated Backpropagation Through Time (TBTT).

    The loss is the standard DDPM noise-prediction MSE (L_diff) plus
    dispersive regularisation on intermediate denoise-MLP features (L_disp).

    Demos are grouped into trajectories by (map, agent) and processed
    sequentially in chunks of ``tbtt_length``, with hidden states detached
    at chunk boundaries to truncate gradient flow.

    When ``checkpoint_every > 0``, activation checkpointing is used within
    each TBTT window: the window is split into segments of
    ``checkpoint_every`` steps and ``torch.utils.checkpoint`` recomputes
    activations during backward instead of storing them.  Gradients still
    flow across the **entire** TBTT window (hidden states are NOT detached
    between checkpoint segments) — only the TBTT boundary performs
    detachment.  This lets you set a very large ``tbtt_length`` (e.g. 7000)
    while keeping peak memory proportional to ``checkpoint_every``.
    """
    from torch.utils.checkpoint import checkpoint as torch_checkpoint

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"\n[Pretrain] Device: {device}")
    ckpt_str = f", checkpoint_every={checkpoint_every}" if checkpoint_every > 0 else ""
    print(f"[Pretrain] {len(demos)} demos, {epochs} epochs, tbtt={tbtt_length}, "
          f"traj_batch={traj_batch_size}{ckpt_str}, lr={lr}")

    # ── Build model ──────────────────────────────────────────────────
    if model_type == "lstm":
        model = DiffusionLSTM(
                    state_dim=STATE_DIM,
                    action_dim=2,
                    num_diffusion_steps=num_diffusion_steps,
                    inference_steps=5,          # DDIM fast sampling for rollout/deploy
                    time_emb_dim=32,
                    hidden_dims=(128, 128),
                    beta_schedule="cosine",
                    odom_expand=32,
                    proj_hidden=384,
                    lstm_hidden_size=128,
                    lstm_num_layers=2,
                    memory_length=64,
                    memory_stride=118
                ).to(device)
    elif model_type == "mamba2":
        model = DiffusionMamba2(
                    state_dim=STATE_DIM,
                    action_dim=2,
                    num_diffusion_steps=num_diffusion_steps,
                    inference_steps=5,          # DDIM fast sampling for rollout/deploy
                    obs_feature_dim=64,
                    time_emb_dim=32,
                    hidden_dims=(128, 128),
                    beta_schedule="cosine",
                    d_model=64,
                    d_state=16,
                    d_conv=4,
                    d_head=16,
                    expand=2,
                    memory_length=6000,
                    odom_expand=32,
                ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Register dispersive hooks on the denoising MLP (last block)
    model.denoise_net.register_dispersive_hooks("late")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ── Build trajectory tensors for TBTT ────────────────────────────
    from collections import defaultdict
    traj_dict = defaultdict(list)
    for d in demos:
        key = (d["map"], d.get("agent", 0))
        traj_dict[key].append(d)
    for key in traj_dict:
        traj_dict[key].sort(key=lambda x: x.get("step", 0))

    traj_data = []  # list of (scans_T, states_T, actions_T) per trajectory
    for key, traj in traj_dict.items():
        if len(traj) < 10:
            continue
        scans  = torch.stack([torch.from_numpy(d["scan"]) for d in traj]).unsqueeze(1).to(device)
        states = torch.stack([torch.from_numpy(d["state"]) for d in traj]).to(device)
        actions = torch.stack([torch.from_numpy(d["action"]) for d in traj]).to(device)
        traj_data.append((scans, states, actions))

    num_trajs = len(traj_data)
    total_steps = sum(t[0].shape[0] for t in traj_data)
    print(f"[Pretrain] {num_trajs} trajectories, {total_steps} total steps, "
          f"TBTT length={tbtt_length}, traj_batch={traj_batch_size}")

    # ── Helper: process a segment of timesteps (used by checkpoint) ──
    def _actor_segment_fn(obs_buffer, hidden_h, hidden_c,
                          scans_seg, states_seg, actions_seg):
        """Forward through a segment, returning (diff, disp, obs_buffer, h, c).
        Captured from closure: model, model_type, dispersive_temperature."""
        seg_len = scans_seg.shape[0]
        seg_diff = torch.tensor(0.0, device=device)
        seg_disp = torch.tensor(0.0, device=device)
        for t in range(seg_len):
            if model_type == "lstm":
                obs_feat, obs_buffer, hidden_h, hidden_c = \
                    model.encode_observation(
                        scans_seg[t], states_seg[t], obs_buffer,
                        hidden_h, hidden_c,
                    )
            else:
                obs_feat, obs_buffer = model.encode_observation(
                    scans_seg[t], states_seg[t], obs_buffer,
                )
            seg_diff = seg_diff + model.compute_diffusion_loss(
                actions_seg[t], obs_feat)
            feat_list = model.denoise_net.get_intermediate_features()
            step_disp = torch.tensor(0.0, device=device)
            if feat_list:
                for feats in feat_list:
                    if feats.ndim > 2:
                        feats = feats.mean(
                            dim=list(range(1, feats.ndim - 1)))
                    step_disp = step_disp + dispersive_loss_infonce_l2(
                        feats, dispersive_temperature)
                step_disp = step_disp / len(feat_list)
            seg_disp = seg_disp + step_disp
        return seg_diff, seg_disp, obs_buffer, hidden_h, hidden_c

    use_ckpt = checkpoint_every > 0

    # ── Training with TBTT ───────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float("inf")
    torch.backends.cudnn.benchmark = True

    for epoch in range(1, epochs + 1):
        model.train()
        traj_order = torch.randperm(num_trajs).tolist()
        epoch_diff = 0.0
        epoch_disp = 0.0
        epoch_steps = 0
        chunk_count = 0
        optimizer.zero_grad(set_to_none=True)

        for tb_start in range(0, num_trajs, traj_batch_size):
            tb_indices = traj_order[tb_start:tb_start + traj_batch_size]
            B = len(tb_indices)
            batch_trajs = [traj_data[i] for i in tb_indices]
            T = min(t[0].shape[0] for t in batch_trajs)

            # Stack trajectories into time-first layout: (T, B, ...)
            scans_tb  = torch.stack([sc[:T] for sc, _, _ in batch_trajs], dim=1)
            states_tb = torch.stack([st[:T] for _, st, _ in batch_trajs], dim=1)
            actions_tb = torch.stack([ac[:T] for _, _, ac in batch_trajs], dim=1)

            # Initialise temporal state for this trajectory batch
            if model_type == "lstm":
                hidden_h, hidden_c = model.get_init_hidden(B, device, transpose=True)
            else:
                hidden_h = torch.empty(0, device=device)
                hidden_c = torch.empty(0, device=device)
            obs_buffer = model.create_observation_buffer(B, device)

            # TBTT: process in chunks, detaching hidden state at boundaries
            for chunk_start in range(0, T, tbtt_length):
                chunk_end = min(chunk_start + tbtt_length, T)

                # Truncate backprop by detaching carried state
                if model_type == "lstm":
                    hidden_h = hidden_h.detach()
                    hidden_c = hidden_c.detach()
                obs_buffer = obs_buffer.detach()

                chunk_diff = torch.tensor(0.0, device=device)
                chunk_disp = torch.tensor(0.0, device=device)
                chunk_len = chunk_end - chunk_start

                if use_ckpt:
                    # ── Activation-checkpointed path ────────────────
                    # Split TBTT window into small segments; gradient
                    # flows across the full window but activations are
                    # recomputed per segment during backward.
                    for seg_start in range(chunk_start, chunk_end,
                                          checkpoint_every):
                        seg_end = min(seg_start + checkpoint_every,
                                      chunk_end)
                        scans_seg   = scans_tb[seg_start:seg_end]
                        states_seg  = states_tb[seg_start:seg_end]
                        actions_seg = actions_tb[seg_start:seg_end]

                        with torch.amp.autocast("cuda",
                                    enabled=(device.type == "cuda")):
                            sd, sp, obs_buffer, hidden_h, hidden_c = \
                                torch_checkpoint(
                                    _actor_segment_fn,
                                    obs_buffer, hidden_h, hidden_c,
                                    scans_seg, states_seg, actions_seg,
                                    use_reentrant=False,
                                    preserve_rng_state=True,
                                )
                        chunk_diff = chunk_diff + sd
                        chunk_disp = chunk_disp + sp
                else:
                    # ── Standard (non-checkpointed) path ────────────
                    with torch.amp.autocast("cuda",
                                enabled=(device.type == "cuda")):
                        for t in range(chunk_start, chunk_end):
                            if model_type == "lstm":
                                obs_feat, obs_buffer, hidden_h, \
                                    hidden_c = \
                                    model.encode_observation(
                                        scans_tb[t], states_tb[t],
                                        obs_buffer, hidden_h, hidden_c,
                                    )
                            else:
                                obs_feat, obs_buffer = \
                                    model.encode_observation(
                                        scans_tb[t], states_tb[t],
                                        obs_buffer,
                                    )
                            diff_loss = model.compute_diffusion_loss(
                                actions_tb[t], obs_feat)
                            feat_list = model.denoise_net \
                                .get_intermediate_features()
                            disp_loss = torch.tensor(0.0, device=device)
                            if feat_list:
                                for feats in feat_list:
                                    if feats.ndim > 2:
                                        feats = feats.mean(
                                            dim=list(range(
                                                1, feats.ndim - 1)))
                                    disp_loss = disp_loss + \
                                        dispersive_loss_infonce_l2(
                                            feats,
                                            dispersive_temperature)
                                disp_loss = disp_loss / len(feat_list)
                            chunk_diff = chunk_diff + diff_loss
                            chunk_disp = chunk_disp + disp_loss

                # Average over timesteps in chunk, scale for grad accumulation
                avg_chunk_loss = (
                    chunk_diff + dispersive_lambda * chunk_disp) / chunk_len
                scaler.scale(
                    avg_chunk_loss / gradient_accumulation_steps).backward()

                chunk_count += 1
                if chunk_count % gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                epoch_diff += chunk_diff.item()
                epoch_disp += chunk_disp.item()
                epoch_steps += chunk_len

            # Progress
            pct = min(tb_start + traj_batch_size, num_trajs) / num_trajs * 100
            n_tb = (num_trajs + traj_batch_size - 1) // traj_batch_size
            print(f"    [Epoch {epoch}/{epochs}] Traj batch "
                  f"{tb_start // traj_batch_size + 1}/{n_tb} "
                  f"({pct:.1f}%)", end="\r", flush=True)

        # Flush remaining accumulated gradients
        if chunk_count % gradient_accumulation_steps != 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        avg_diff = epoch_diff / max(epoch_steps, 1)
        avg_disp = epoch_disp / max(epoch_steps, 1)
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
# Critic pre-training  (uses same demos, proxy reward → MC returns)
# ──────────────────────────────────────────────────────────────────────

def pretrain_critic(
    demos: list[dict],
    actor_state_dict: dict,
    model_type: str = "lstm",
    epochs: int = 30,
    tbtt_length: int = 64,
    checkpoint_every: int = 0,
    traj_batch_size: int = 8,
    lr: float = 5e-4,
    gamma: float = 0.999,
    speed_reward_scale: float = 0.10,
    collision_penalty: float = -4.0,
    save_dir: str = "models/critic/pretrained",
    save_name: str = "critic_pretrained.pt",
    device: str | torch.device = "cuda",
):
    """
    Pre-train the CriticNetwork value function on MC returns computed from
    a proxy reward derived from the same expert demonstrations used for
    actor pretraining.

    Proxy reward per transition:
        r = next_velocity * speed_reward_scale
            + next_collision * collision_penalty

    The critic's CNN (vision encoder) is initialised from the pretrained
    actor weights and frozen; only the LSTM, projection, and value-head
    layers are trained.

    Falls back to ``state[0]`` (vel_x) when demos lack ``next_velocity``
    (backward compat with older demo files).
    """
    from models.CriticNetworks import CriticNetwork, Mamba2CriticNetwork
    from models.AuxModels import VisionEncoder

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 60}")
    print("  CRITIC PRE-TRAINING  (weight_initializer)")
    print(f"{'=' * 60}")
    print(f"  {len(demos)} demos, {epochs} epochs, tbtt={tbtt_length}, "
          f"traj_batch={traj_batch_size}, "
          f"{'ckpt=' + str(checkpoint_every) + ', ' if checkpoint_every > 0 else ''}"
          f"lr={lr}, γ={gamma}")

    has_reward_fields = ("next_velocity" in demos[0])
    if not has_reward_fields:
        print("  ⚠ Demos lack next_velocity/next_collision — using state vel_x proxy.")

    # ── 1. Build CriticNetwork with pretrained vision encoder ────────
    encoder = VisionEncoder(num_scan_beams=NUM_BEAMS)
    # Transfer actor CNN weights to critic encoder
    prefixes = ["conv_layers.", "vision_encoder.",
                "0.module.conv_layers.", "0.module.vision_encoder."]
    encoder_sd = {}
    for k, v in actor_state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        for prefix in prefixes:
            if k.startswith(prefix):
                encoder_sd[k[len(prefix):]] = v
                break
    if encoder_sd:
        ref_sd = encoder.state_dict()
        filtered = {k: v for k, v in encoder_sd.items()
                    if k in ref_sd and ref_sd[k].shape == v.shape}
        if filtered:
            encoder.load_state_dict(filtered, strict=False)
            print(f"  Loaded {len(filtered)}/{len(ref_sd)} actor CNN weights "
                  f"into critic encoder.")

    if model_type == "lstm":
        critic = CriticNetwork(
            state_dim=STATE_DIM,
            encoder=encoder,
            lstm_hidden_size=64,
            lstm_num_layers=2,
            memory_length=64,
            memory_stride=118,
            odom_expand=64,
            proj_hidden=256,
        ).to(device)
    elif model_type == "mamba2":
        critic = Mamba2CriticNetwork(
            state_dim=STATE_DIM,
            encoder=encoder,
            d_model=64,
            d_state=16,
            d_conv=4,
            d_head=16,
            expand=2,
            memory_length=7500,
            odom_expand=64,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Freeze vision encoder (already pretrained)
    for p in critic.conv_layers.parameters():
        p.requires_grad = False

    # ── 2. Reconstruct trajectories & compute MC returns ─────────────
    # Group demos by (map, agent) to get per-agent time-ordered sequences.
    from collections import defaultdict
    trajectories = defaultdict(list)
    for d in demos:
        key = (d["map"], d.get("agent", 0))
        trajectories[key].append(d)

    # Sort each trajectory by step index
    for key in trajectories:
        trajectories[key].sort(key=lambda x: x.get("step", 0))

    traj_data_c = []  # list of (scans, states, returns) per trajectory

    for (map_name, agent_idx), traj in trajectories.items():
        T = len(traj)
        if T < 10:
            continue

        scans  = torch.stack([torch.from_numpy(d["scan"]) for d in traj]).unsqueeze(1).to(device)
        states = torch.stack([torch.from_numpy(d["state"]) for d in traj]).to(device)

        # Compute per-step rewards
        rewards = torch.zeros(T)
        for t, d in enumerate(traj):
            if has_reward_fields:
                vel = d["next_velocity"]
                col = d["next_collision"]
            else:
                vel = float(d["state"][0])   # vel_x as proxy
                col = 0
            rewards[t] = vel * speed_reward_scale + col * collision_penalty

        # Normalise rewards per trajectory
        r_std = rewards.std().clamp(min=1e-4)
        rewards = rewards / r_std

        # MC returns (reverse cumsum)
        returns = torch.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + gamma * G
            returns[t] = G

        traj_data_c.append((scans, states, returns.to(device)))

    num_trajs_c = len(traj_data_c)
    total_steps_c = sum(t[0].shape[0] for t in traj_data_c)
    print(f"  {num_trajs_c} trajectories, {total_steps_c} steps, "
          f"TBTT length={tbtt_length}, traj_batch={traj_batch_size}")

    # ── 3. TBTT training ────────────────────────────────────────────
    from torch.utils.checkpoint import checkpoint as torch_checkpoint

    def _critic_segment_fn(obs_buffer, hidden_h, hidden_c,
                           scans_seg, states_seg, returns_seg):
        """Forward through a critic segment.
        Captured from closure: critic, model_type."""
        seg_len = scans_seg.shape[0]
        seg_loss = torch.tensor(0.0, device=scans_seg.device)
        for t in range(seg_len):
            if model_type == "lstm":
                feat, obs_buffer, hidden_h, hidden_c = \
                    critic.encode_observation(
                        scans_seg[t], states_seg[t], obs_buffer,
                        hidden_h, hidden_c,
                    )
            else:
                feat, obs_buffer = critic.encode_observation(
                    scans_seg[t], states_seg[t], obs_buffer,
                )
            pred = critic.fc_layers(feat).squeeze(-1)
            seg_loss = seg_loss + F.smooth_l1_loss(pred, returns_seg[t])
        return seg_loss, obs_buffer, hidden_h, hidden_c

    use_ckpt = checkpoint_every > 0

    critic.train()
    optim_c = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, critic.parameters()),
        lr=lr, weight_decay=0.01,
    )

    best_loss = float("inf")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        traj_order = torch.randperm(num_trajs_c).tolist()
        epoch_loss = 0.0
        epoch_steps = 0

        for tb_start in range(0, num_trajs_c, traj_batch_size):
            tb_indices = traj_order[tb_start:tb_start + traj_batch_size]
            B = len(tb_indices)
            batch_trajs = [traj_data_c[i] for i in tb_indices]
            T = min(t[0].shape[0] for t in batch_trajs)

            # Stack into time-first layout
            scans_tb   = torch.stack([sc[:T] for sc, _, _ in batch_trajs], dim=1)
            states_tb  = torch.stack([st[:T] for _, st, _ in batch_trajs], dim=1)
            returns_tb = torch.stack([ret[:T] for _, _, ret in batch_trajs], dim=1)

            # Initialise temporal state
            if model_type == "lstm":
                hidden_h, hidden_c = critic.get_init_hidden(B, device, transpose=True)
            else:
                hidden_h = torch.empty(0, device=device)
                hidden_c = torch.empty(0, device=device)
            obs_buffer = critic.create_observation_buffer(B, device)

            # Process in TBTT chunks
            for chunk_start in range(0, T, tbtt_length):
                chunk_end = min(chunk_start + tbtt_length, T)

                if model_type == "lstm":
                    hidden_h = hidden_h.detach()
                    hidden_c = hidden_c.detach()
                obs_buffer = obs_buffer.detach()

                chunk_loss = torch.tensor(0.0, device=device)
                chunk_len = chunk_end - chunk_start

                if use_ckpt:
                    for seg_start in range(chunk_start, chunk_end,
                                          checkpoint_every):
                        seg_end = min(seg_start + checkpoint_every,
                                      chunk_end)
                        sl, obs_buffer, hidden_h, hidden_c = \
                            torch_checkpoint(
                                _critic_segment_fn,
                                obs_buffer, hidden_h, hidden_c,
                                scans_tb[seg_start:seg_end],
                                states_tb[seg_start:seg_end],
                                returns_tb[seg_start:seg_end],
                                use_reentrant=False,
                                preserve_rng_state=True,
                            )
                        chunk_loss = chunk_loss + sl
                else:
                    for t in range(chunk_start, chunk_end):
                        if model_type == "lstm":
                            feat, obs_buffer, hidden_h, hidden_c = \
                                critic.encode_observation(
                                    scans_tb[t], states_tb[t],
                                    obs_buffer, hidden_h, hidden_c,
                                )
                        else:
                            feat, obs_buffer = critic.encode_observation(
                                scans_tb[t], states_tb[t], obs_buffer,
                            )
                        pred = critic.fc_layers(feat).squeeze(-1)
                        chunk_loss = chunk_loss + F.smooth_l1_loss(
                            pred, returns_tb[t])

                avg_loss = chunk_loss / chunk_len
                optim_c.zero_grad(set_to_none=True)
                avg_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                optim_c.step()

                epoch_loss += chunk_loss.item()
                epoch_steps += chunk_len

        avg = epoch_loss / max(epoch_steps, 1)
        if avg < best_loss:
            best_loss = avg
            torch.save(critic.state_dict(), os.path.join(save_dir, save_name))
            marker = " ★"
        else:
            marker = ""
        if epoch % max(1, epochs // 20) == 0 or epoch <= 3 or marker:
            print(f"  Epoch {epoch:>3d}/{epochs}  loss={avg:.5f}  best={best_loss:.5f}{marker}")

    # Final save
    torch.save(critic.state_dict(), os.path.join(save_dir, "critic_final.pt"))
    print(f"\n  Critic pretraining done.  Best loss: {best_loss:.5f}")
    print(f"  Saved to {os.path.join(save_dir, save_name)}")
    print("=" * 60 + "\n")
    return critic


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    ALL_MAPS = [
        "Hockenheim", "Monza", "Melbourne", "BrandsHatch",
        "Oschersleben", "Sakhir", "Sepang", "SaoPaulo",
        "Budapest", "Catalunya", "Silverstone",
        "Zandvoort", "MoscowRaceway", "Nuerburgring",
        "Sochi",
    ]

    parser = argparse.ArgumentParser(description="D²PPO Stage 1: BC Pre-training from MPC")
    # Demo collection
    parser.add_argument("--maps", nargs="+", default=ALL_MAPS,
                        help="Maps to collect demos on (default: all)")
    parser.add_argument("--num_agents", type=int, default=2,
                        help="Agents per env during collection")
    parser.add_argument("--steps_per_map", type=int, default=7000,
                        help="Env steps per map during collection")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Parallel processes for demo collection (default: min(num_maps, cpu_count))")
    parser.add_argument("--load", type=str, default=None,
                        help="Path to a saved demos .pt file (skip collection)")
    parser.add_argument("--save_demos", type=str, default="demos/expert_demos.pt",
                        help="Where to save collected demos for reuse")
    # Actor training
    parser.add_argument("--model_type", type=str, default="lstm", choices=["lstm", "mamba2"],
                        help="Network architecture backbone (lstm or mamba2)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--tbtt_length", type=int, default=512,
                        help="TBTT truncation window length")
    parser.add_argument("--checkpoint_every", type=int, default=0,
                        help="Activation checkpoint segment size inside TBTT window "
                             "(0=off, e.g. 64 for memory saving with large tbtt_length)")
    parser.add_argument("--traj_batch_size", type=int, default=2,
                        help="Number of trajectories to process in parallel")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dispersive_lambda", type=float, default=0.5)
    parser.add_argument("--num_diffusion_steps", type=int, default=10)
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    # Actor output
    parser.add_argument("--save_dir", type=str, default="models/actor/pretrained")
    parser.add_argument("--save_name", type=str, default="actor_pretrained.pt")
    # Critic pretraining
    parser.add_argument("--critic_epochs", type=int, default=30)
    parser.add_argument("--critic_lr", type=float, default=5e-4)
    parser.add_argument("--critic_save_dir", type=str, default="models/critic/pretrained")
    parser.add_argument("--critic_save_name", type=str, default="critic_pretrained.pt")
    parser.add_argument("--skip_critic", action="store_true",
                        help="Skip critic pretraining (actor only)")

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

    # ── Train actor ──────────────────────────────────────────────────
    model = pretrain(
        demos=demos,
        model_type=args.model_type,
        epochs=args.epochs,
        tbtt_length=args.tbtt_length,
        checkpoint_every=args.checkpoint_every,
        traj_batch_size=args.traj_batch_size,
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

        if args.model_type == "lstm":
            obs_feat, _, _, _ = model.encode_observation(scans, states, obs_buffer=None)
        else:
            obs_feat, _ = model.encode_observation(scans, states, obs_buffer=None)
            
        sampled = model.sample_action(obs_feat, deterministic=True)

        print(f"  Expert actions (first 4): {expert[:4].cpu().numpy()}")
        print(f"  Model actions  (first 4): {sampled[:4].cpu().numpy()}")
        mse = F.mse_loss(sampled, expert).item()
        print(f"  Action MSE: {mse:.5f}")

    # ── Train critic (reuse same demos) ──────────────────────────────
    actor_weights_path = os.path.join(args.save_dir, args.save_name)
    if not args.skip_critic:
        actor_sd = torch.load(actor_weights_path, weights_only=False)
        pretrain_critic(
            demos=demos,
            actor_state_dict=actor_sd,
            model_type=args.model_type,
            epochs=args.critic_epochs,
            tbtt_length=args.tbtt_length,
            checkpoint_every=args.checkpoint_every,
            traj_batch_size=args.traj_batch_size,
            lr=args.critic_lr,
            save_dir=args.critic_save_dir,
            save_name=args.critic_save_name,
        )
        critic_path = os.path.join(args.critic_save_dir, args.critic_save_name)
        print("\n✓ Pre-training complete. Use the saved weights with D2PPO_agent via:")
        print(f'    transfer=["{actor_weights_path}", "{critic_path}"]')
    else:
        print("\n✓ Actor pre-training complete (critic skipped). Use with:")
        print(f'    transfer=["{actor_weights_path}", None]')


if __name__ == "__main__":
    main()
