# -*- coding: utf-8 -*-
"""
weight_initializer.py — Behavioral Cloning Pre-training for DiffusionLSTM
==========================================================================
Collects expert demonstrations from either:
  1. MPC baseline (default)
  2. A pretrained model from racing_rl (--pretrained_expert)

Then trains the DiffusionLSTM/Mamba2 diffusion actor using
the D²PPO Stage-1 objective:

    L = L_diff + λ · L_disp           (Eq. 6, Zou et al. 2025)

Usage:
    python weight_initializer.py                        # MPC expert (default)
    python weight_initializer.py --pretrained_expert /path/to/actor_val_best.pt
    python weight_initializer.py --epochs 200 --maps Hockenheim Monza
    python weight_initializer.py --load demos.pt        # skip collection, train from saved demos

Outputs:
    models/actor/pretrained/actor_pretrained.pt   — pretrained actor weights
    demos/expert_demos.pt                         — collected demonstrations (reusable)
"""

import argparse
import os
import sys
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
from utils.diffusion_utils import extract, dispersive_loss_infonce_l2
from baselines.mpc_agent import MPCAgent
from utils.sim_config import D2PPO_STATE_DIM, LIDAR_BEAMS, LIDAR_FOV, SIM_PARAMS
from utils.utils import get_map_dir, generate_start_poses

# Add racing_rl to path for loading pretrained models
RACING_RL_PATH = "/home/WVU-AD/ads00024/racing_rl"
if RACING_RL_PATH not in sys.path:
    sys.path.insert(0, RACING_RL_PATH)


PARAMS_DICT = SIM_PARAMS.copy()
NUM_BEAMS = LIDAR_BEAMS
STATE_DIM = D2PPO_STATE_DIM


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

        # Determine which agents will be reset this step
        stuck = (collisions == 1) | (np.abs(velocities) < 0.1)
        collision_timers[stuck] += 1
        collision_timers[~stuck] = 0
        to_reset = np.where(collision_timers >= collision_reset_threshold)[0]
        reset_set = set(to_reset.tolist()) if len(to_reset) > 0 else set()

        for a_idx in range(num_agents):
            scan_np = np.asarray(obs["scans"][a_idx], dtype=np.float32)
            state_np = np.array([
                obs["linear_vels_x"][a_idx],
                obs["linear_vels_y"][a_idx],
                obs["ang_vels_z"][a_idx],
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
                "done": int(a_idx in reset_set),
            })

        # Execute reset
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


def _env_worker_fn(conn, map_name, num_agents, num_noise_agents,
                   collision_reset_threshold):
    """
    Subprocess: owns one gym env + MPC noise controller.
    Receives expert actions via *conn*, steps the env, sends back
    observations and demo metadata.  Runs on its own CPU core.
    """
    import gym as _gym

    total_agents = num_agents + num_noise_agents
    SIM_DT = 0.01

    env = _gym.make(
        "f110_gym:f110-v0",
        map=get_map_dir(map_name) + f"/{map_name}_map",
        num_agents=total_agents,
        num_beams=NUM_BEAMS,
        fov=LIDAR_FOV,
        params=PARAMS_DICT,
    )

    noise_mpc = MPCAgent(
        map_name=map_name,
        wheelbase=PARAMS_DICT['lf'] + PARAMS_DICT['lr'],
        max_steering=PARAMS_DICT['s_max'],
        max_accel=PARAMS_DICT['a_max'],
        LIDAR_FOV=LIDAR_FOV,
        num_beams=NUM_BEAMS,
        horizon=8, speed_scale=0.8,
        emergency_dist=0.8, speed_clamp=7.5,
    )

    poses = generate_start_poses(map_name, total_agents)
    obs, _, _, _ = env.reset(poses=poses)

    collision_timers = np.zeros(total_agents, dtype=np.int32)
    prev_vels_x = np.zeros(total_agents, dtype=np.float32)

    def _build_obs_arrays(obs_dict):
        """Extract numpy arrays from obs dict for pipe transport."""
        nonlocal prev_vels_x
        scans = np.array(obs_dict["scans"], dtype=np.float32)
        lvx = np.array(obs_dict["linear_vels_x"], dtype=np.float32)
        s3d = np.stack([lvx, obs_dict["linear_vels_y"],
                        obs_dict["ang_vels_z"]], axis=1).astype(np.float32)
        accel = np.clip((lvx - prev_vels_x) / SIM_DT, -10.0, 10.0)
        s4d = np.column_stack([s3d, accel]).astype(np.float32)
        prev_vels_x = lvx.copy()
        return scans, s3d, s4d

    # Send initial observations to main process
    scans, s3d, s4d = _build_obs_arrays(obs)
    conn.send(("init", scans, s3d, s4d))

    while True:
        msg = conn.recv()
        if msg is None:  # shutdown
            break

        expert_actions = msg  # (total_agents, 2)

        # Apply MPC noise for background traffic
        noise_actions = noise_mpc.get_actions_batch(obs)
        actions = expert_actions.copy()
        actions[num_agents:] = noise_actions[:num_noise_agents]
        actions[-3:] = np.array([0.0, 0.0])

        next_obs, _, _, _ = env.step(actions)

        # Collision / stuck detection
        collisions = np.array(next_obs["collisions"])
        velocities = np.array(next_obs["linear_vels_x"])
        stuck = (collisions == 1) | (np.abs(velocities) < 0.1)
        collision_timers[stuck] += 1
        collision_timers[~stuck] = 0
        to_reset = np.where(
            collision_timers >= collision_reset_threshold)[0]

        # Demo metadata (actions taken, next-step reward info)
        demo_meta = (
            actions[:num_agents].astype(np.float32).copy(),
            velocities[:num_agents].astype(np.float32).copy(),
            collisions[:num_agents].astype(np.int32).copy(),
            to_reset[to_reset < num_agents].tolist(),
        )

        # Reset stuck agents
        reset_all = to_reset.tolist()
        if len(to_reset) > 0:
            cur_poses = np.stack([
                next_obs["poses_x"], next_obs["poses_y"],
                next_obs["poses_theta"],
            ], axis=1)
            new_poses = generate_start_poses(
                map_name, total_agents, agent_poses=cur_poses)
            next_obs, _, _, _ = env.reset(
                poses=new_poses, agent_idxs=to_reset)
            collision_timers[to_reset] = 0
            prev_vels_x[to_reset] = 0.0

        obs = next_obs
        scans, s3d, s4d = _build_obs_arrays(obs)
        conn.send(("step", scans, s3d, s4d, demo_meta, reset_all))

    env.close()
    conn.close()


def _collect_pretrained_batched(
    map_names: list[str],
    pretrained_model_path: str,
    num_agents: int,
    steps_per_map: int,
    collision_reset_threshold: int,
    num_noise_agents: int = 16,
    verbose: bool = True,
) -> list[dict]:
    """
    Run *len(map_names)* gym environments in **separate subprocesses**
    (one per CPU core), batching all agents into a single GPU forward
    pass of the pretrained expert model each step.

    Architecture:
        Main process (GPU) ←→ Worker 0 (CPU core, env 0)
                            ←→ Worker 1 (CPU core, env 1)
                            ←→ Worker 2 (CPU core, env 2)

    Each step:
        1. Main gathers obs from workers (already received)
        2. Batched GPU forward pass → expert actions
        3. Send actions to all workers (they step in parallel)
        4. Receive results from all workers
    """
    import sys

    if RACING_RL_PATH not in sys.path:
        sys.path.insert(0, RACING_RL_PATH)
    from model import ExampleNetwork, VisionEncoder as RacingVisionEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    E = len(map_names)
    total_agents = num_agents + num_noise_agents
    N_total = E * total_agents

    ctx = mp.get_context("fork")
    parent_conns = []
    workers = []
    for map_name in map_names:
        parent_conn, child_conn = ctx.Pipe()
        p = ctx.Process(
            target=_env_worker_fn,
            args=(child_conn, map_name, num_agents, num_noise_agents,
                  collision_reset_threshold),
            daemon=True,
        )
        p.start()
        child_conn.close()  # parent doesn't use child end
        parent_conns.append(parent_conn)
        workers.append(p)

    # Receive initial observations from all workers
    all_scans = []   # list of (total_agents, beams) per env
    all_s3d = []     # list of (total_agents, 3) per env
    all_s4d = []     # list of (total_agents, 4) per env
    for conn in parent_conns:
        tag, scans, s3d, s4d = conn.recv()
        all_scans.append(scans)
        all_s3d.append(s3d)
        all_s4d.append(s4d)

    encoder = RacingVisionEncoder(num_scan_beams=NUM_BEAMS)
    expert_model = ExampleNetwork(
        state_dim=4, action_dim=2, encoder=encoder,
        lstm_hidden_size=512, lstm_num_layers=2,
        memory_length=48, memory_stride=1,
    ).to(device)

    checkpoint = torch.load(pretrained_model_path, map_location=device,
                            weights_only=False)
    if isinstance(checkpoint, dict):
        state_dict = {}
        for k, v in checkpoint.items():
            if not isinstance(v, torch.Tensor):
                continue
            clean_k = k
            for prefix in ("_orig_mod.", "0.module."):
                if clean_k.startswith(prefix):
                    clean_k = clean_k[len(prefix):]
            state_dict[clean_k] = v
        result = expert_model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            print(f"  [Warning] Missing keys: {result.missing_keys[:5]}...")
        if result.unexpected_keys:
            print(f"  [Warning] Unexpected keys: "
                  f"{result.unexpected_keys[:5]}...")
    expert_model.eval()
    if verbose:
        print(f"  [Pretrained] Loaded expert — {E} env workers "
              f"({N_total} agents batched, {E} CPU cores)")

    obs_buffer = expert_model.create_observation_buffer(N_total, device)
    hidden_h, hidden_c = expert_model.get_init_hidden(
        N_total, device, transpose=True)

    demos: list[dict] = []

    for step in range(steps_per_map):
        batch_scans = np.concatenate(all_scans, axis=0)
        batch_s4d = np.concatenate(all_s4d, axis=0)

        scan_t = torch.from_numpy(batch_scans).unsqueeze(1).to(device)
        state_t = torch.from_numpy(batch_s4d).to(device)

        with torch.no_grad():
            loc, _, obs_buffer, hidden_h, hidden_c = expert_model(
                scan_t, state_t, obs_buffer, hidden_h, hidden_c)
            expert_actions_all = loc.cpu().numpy()

        for e, conn in enumerate(parent_conns):
            off = e * total_agents
            conn.send(expert_actions_all[off:off + total_agents])

        new_scans, new_s3d, new_s4d = [], [], []
        for e, conn in enumerate(parent_conns):
            tag, scans, s3d, s4d, demo_meta, reset_all = conn.recv()
            acts, vels, cols, reset_demo = demo_meta
            off = e * total_agents

            for a_idx in range(num_agents):
                demos.append({
                    "scan":   all_scans[e][a_idx].copy(),
                    "state":  all_s3d[e][a_idx].copy(),
                    "action": acts[a_idx],
                    "map":    map_names[e],
                    "step":   step,
                    "agent":  a_idx,
                    "next_velocity":  float(vels[a_idx]),
                    "next_collision": int(cols[a_idx]),
                    "done": int(a_idx in reset_demo),
                })

            for idx in reset_all:
                g = off + idx
                obs_buffer[g] = 0.0
                hidden_h[g] = 0.0
                hidden_c[g] = 0.0

            new_scans.append(scans)
            new_s3d.append(s3d)
            new_s4d.append(s4d)

        all_scans = new_scans
        all_s3d = new_s3d
        all_s4d = new_s4d

        if verbose and step % 500 == 0:
            print(f"    step {step}/{steps_per_map}  "
                  f"demos={len(demos)}", flush=True)

    for conn in parent_conns:
        conn.send(None)
        conn.close()
    for p in workers:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

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


def collect_demos_pretrained(
    maps: list[str],
    pretrained_model_path: str,
    num_agents: int = 4,
    steps_per_map: int = 2000,
    collision_reset_threshold: int = 32,
    concurrent_envs: int = 3,
    verbose: bool = True,
) -> list[dict]:
    """
    Collect demonstrations using a pretrained RecurrentActorNetwork from racing_rl
    across all *maps*.

    Runs ``concurrent_envs`` gym environments simultaneously, batching all
    agents into a single GPU forward pass per step for ~Nx speedup.
    Maps are processed in groups of ``concurrent_envs``.
    """
    if verbose:
        print(f"\n[Demo] Collecting with pretrained model on {len(maps)} maps "
              f"({num_agents} agents × {steps_per_map} steps each, "
              f"{concurrent_envs} concurrent envs)")
        print(f"       Model: {pretrained_model_path}")

    demos: list[dict] = []

    # Process maps in batches of concurrent_envs
    for batch_start in range(0, len(maps), concurrent_envs):
        batch_maps = maps[batch_start:batch_start + concurrent_envs]
        try:
            batch_demos = _collect_pretrained_batched(
                batch_maps, pretrained_model_path, num_agents,
                steps_per_map, collision_reset_threshold,
                verbose=verbose,
            )
            demos.extend(batch_demos)
            if verbose:
                print(f"  ✓ batch {batch_maps}: {len(batch_demos)} demos  "
                      f"(total so far: {len(demos)})")
        except Exception as exc:
            import traceback
            print(f"  ✗ batch {batch_maps} failed: {exc}")
            traceback.print_exc()

    if verbose:
        print(f"\n[Demo] Total demonstrations collected: {len(demos)}")
    return demos


def pretrain(
    demos: list[dict],
    model_type: str = "lstm",
    epochs: int = 100,
    tbtt_length: int = 64,
    traj_batch_size: int = 8,
    lr: float = 3e-4,
    dispersive_lambda: float = 0.5,
    dispersive_temperature: float = 0.5,
    num_diffusion_steps: int = 100,
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

    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"\n[Pretrain] Device: {device}")
    print(f"[Pretrain] {len(demos)} demos, {epochs} epochs, tbtt={tbtt_length}, "
          f"traj_batch={traj_batch_size}, lr={lr}")

    if model_type == "lstm":
        model = DiffusionLSTM(
                    state_dim=STATE_DIM,
                    action_dim=2,
                    num_diffusion_steps=num_diffusion_steps,
                    inference_steps=0,          # DDIM fast sampling for rollout/deploy
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
                    inference_steps=0,          # DDIM fast sampling for rollout/deploy
                    obs_feature_dim=256,
                    time_emb_dim=32,
                    hidden_dims=(256, 256),
                    beta_schedule="cosine",
                    d_model=256,
                    d_state=128,
                    d_conv=4,
                    d_head=32,
                    expand=2,
                    odom_expand=64,
                ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.denoise_net.register_dispersive_hooks("late")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    from collections import defaultdict
    traj_dict = defaultdict(list)
    for d in demos:
        key = (d["map"], d.get("agent", 0))
        traj_dict[key].append(d)
    for key in traj_dict:
        traj_dict[key].sort(key=lambda x: x.get("step", 0))

    traj_data = []  # list of (scans_T, states_T, actions_T, dones_T) per trajectory
    for key, traj in traj_dict.items():
        if len(traj) < 10:
            continue
        scans  = torch.stack([torch.from_numpy(d["scan"]) for d in traj]).unsqueeze(1).to(device)
        states = torch.stack([torch.from_numpy(d["state"]) for d in traj]).to(device)
        actions = torch.stack([torch.from_numpy(d["action"]) for d in traj]).to(device)
        dones  = torch.tensor([d.get("done", 0) for d in traj], dtype=torch.float32, device=device)
        traj_data.append((scans, states, actions, dones))

    num_trajs = len(traj_data)
    total_steps = sum(t[0].shape[0] for t in traj_data)
    print(f"[Pretrain] {num_trajs} trajectories, {total_steps} total steps, "
          f"tbtt={tbtt_length}, traj_batch={traj_batch_size}")

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

            scans_tb  = torch.stack([sc[:T] for sc, _, _, _ in batch_trajs], dim=1)
            states_tb = torch.stack([st[:T] for _, st, _, _ in batch_trajs], dim=1)
            actions_tb = torch.stack([ac[:T] for _, _, ac, _ in batch_trajs], dim=1)
            dones_tb  = torch.stack([dn[:T] for _, _, _, dn in batch_trajs], dim=1)  # (T, B)

            effective_tbtt = min(tbtt_length, T)

            if model_type == "lstm":
                hidden_h, hidden_c = model.get_init_hidden(B, device, transpose=True)
                obs_buffer = model.create_observation_buffer(B, device)
            else:
                conv_state, ssm_state = model.allocate_state(B, device)

            for chunk_start in range(0, T, effective_tbtt):
                chunk_end = min(chunk_start + effective_tbtt, T)

                if model_type == "lstm":
                    hidden_h = hidden_h.detach()
                    hidden_c = hidden_c.detach()
                    obs_buffer = obs_buffer.detach().requires_grad_()
                else:
                    conv_state = conv_state.detach()
                    ssm_state = ssm_state.detach()

                chunk_diff = torch.tensor(0.0, device=device)
                chunk_disp = torch.tensor(0.0, device=device)
                chunk_len = chunk_end - chunk_start

                for t in range(chunk_start, chunk_end):
                    # Reset trajectories that ended at the previous step.
                    if t > 0 and dones_tb[t - 1].any():
                        reset_idx = dones_tb[t - 1].nonzero(
                            as_tuple=False).squeeze(-1)
                        if model_type == "lstm":
                            hidden_h[reset_idx] = 0.0
                            hidden_c[reset_idx] = 0.0
                            obs_buffer[reset_idx] = 0.0
                        else:
                            conv_state[reset_idx] = 0.0
                            ssm_state[reset_idx] = 0.0

                    if model_type == "lstm":
                        obs_feat, obs_buffer, hidden_h, \
                            hidden_c = \
                            model.encode_observation(
                                scans_tb[t], states_tb[t],
                                obs_buffer, hidden_h, hidden_c,
                            )
                    else:
                        obs_feat, conv_state, ssm_state = \
                            model.encode_observation(
                                scans_tb[t], states_tb[t],
                                conv_state, ssm_state,
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
                (avg_chunk_loss / gradient_accumulation_steps).backward()

                chunk_count += 1
                if chunk_count % gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
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
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
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

    torch.save(model.state_dict(), os.path.join(save_dir, "actor_final.pt"))
    print(f"\n[Pretrain] Done.  Best total loss: {best_loss:.5f}")
    print(f"  Saved to {os.path.join(save_dir, save_name)}")
    return model


def pretrain_critic(
    demos: list[dict],
    actor_state_dict: dict,
    model_type: str = "lstm",
    epochs: int = 30,
    tbtt_length: int = 64,
    traj_batch_size: int = 8,
    lr: float = 5e-4,
    gamma: float = 0.99,
    speed_reward_scale: float = 0.10,
    collision_penalty: float = -5.0,
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
          f"traj_batch={traj_batch_size}, lr={lr}, γ={gamma}")

    has_reward_fields = ("next_velocity" in demos[0])
    if not has_reward_fields:
        print("  ⚠ Demos lack next_velocity/next_collision — using state vel_x proxy.")

    encoder = VisionEncoder(num_scan_beams=NUM_BEAMS)
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
            d_model=256,
            d_state=128,
            d_conv=4,
            d_head=32,
            expand=2,
            odom_expand=64,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    for p in critic.conv_layers.parameters():
        p.requires_grad = False

    from collections import defaultdict
    trajectories = defaultdict(list)
    for d in demos:
        key = (d["map"], d.get("agent", 0))
        trajectories[key].append(d)

    for key in trajectories:
        trajectories[key].sort(key=lambda x: x.get("step", 0))

    traj_data_c = []  # list of (scans, states, returns, dones) per trajectory

    for (map_name, agent_idx), traj in trajectories.items():
        T = len(traj)
        if T < 10:
            continue

        scans  = torch.stack([torch.from_numpy(d["scan"]) for d in traj]).unsqueeze(1).to(device)
        states = torch.stack([torch.from_numpy(d["state"]) for d in traj]).to(device)
        dones  = torch.tensor([d.get("done", 0) for d in traj], dtype=torch.float32, device=device)

        rewards = torch.zeros(T)
        for t, d in enumerate(traj):
            if has_reward_fields:
                vel = d["next_velocity"]
                col = d["next_collision"]
            else:
                vel = float(d["state"][0])   # vel_x as proxy
                col = 0
            rewards[t] = vel * speed_reward_scale + col * collision_penalty

        r_mean = rewards.mean()
        r_std = rewards.std().clamp(min=1e-4)
        rewards = (rewards - r_mean) / r_std

        returns = torch.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t] + gamma * G * (1.0 - dones[t].item())
            returns[t] = G

        traj_data_c.append((scans, states, returns.to(device), dones))

    num_trajs_c = len(traj_data_c)
    total_steps_c = sum(t[0].shape[0] for t in traj_data_c)
    print(f"  {num_trajs_c} trajectories, {total_steps_c} steps, "
          f"tbtt={tbtt_length}, traj_batch={traj_batch_size}")

    critic.train()
    optim_c = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, critic.parameters()),
        lr=lr, weight_decay=0,
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

            scans_tb   = torch.stack([sc[:T] for sc, _, _, _ in batch_trajs], dim=1)
            states_tb  = torch.stack([st[:T] for _, st, _, _ in batch_trajs], dim=1)
            returns_tb = torch.stack([ret[:T] for _, _, ret, _ in batch_trajs], dim=1)
            dones_tb   = torch.stack([dn[:T] for _, _, _, dn in batch_trajs], dim=1)  # (T, B)

            effective_tbtt = min(tbtt_length, T)

            if model_type == "lstm":
                hidden_h, hidden_c = critic.get_init_hidden(B, device, transpose=True)
                obs_buffer = critic.create_observation_buffer(B, device)
            else:
                conv_state, ssm_state = critic.allocate_state(B, device)

            for chunk_start in range(0, T, effective_tbtt):
                chunk_end = min(chunk_start + effective_tbtt, T)

                if model_type == "lstm":
                    hidden_h = hidden_h.detach()
                    hidden_c = hidden_c.detach()
                    obs_buffer = obs_buffer.detach().requires_grad_()
                else:
                    conv_state = conv_state.detach()
                    ssm_state = ssm_state.detach()

                chunk_loss = torch.tensor(0.0, device=device)
                chunk_len = chunk_end - chunk_start

                for t in range(chunk_start, chunk_end):
                    # Reset temporal state for trajectories that
                    # had a done at the previous step.
                    if t > 0 and dones_tb[t - 1].any():
                        reset_idx = dones_tb[t - 1].nonzero(
                            as_tuple=False).squeeze(-1)
                        if model_type == "lstm":
                            hidden_h[reset_idx] = 0.0
                            hidden_c[reset_idx] = 0.0
                            obs_buffer[reset_idx] = 0.0
                        else:
                            conv_state[reset_idx] = 0.0
                            ssm_state[reset_idx] = 0.0

                    if model_type == "lstm":
                        feat, obs_buffer, hidden_h, hidden_c = \
                            critic.encode_observation(
                                scans_tb[t], states_tb[t],
                                obs_buffer, hidden_h, hidden_c,
                            )
                    else:
                        feat, conv_state, ssm_state = \
                            critic.encode_observation(
                                scans_tb[t], states_tb[t],
                                conv_state, ssm_state,
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

def main():
    ALL_MAPS = [
        "Hockenheim", "Monza", "Melbourne", "BrandsHatch",
        "Oschersleben", "Sakhir", "Sepang", "SaoPaulo",
        "Budapest", "Catalunya", "Silverstone",
        "Zandvoort", "MoscowRaceway", "Nuerburgring",
        "Sochi",
    ]

    # Maps the racing_rl SupervisedAgent has completed at least 1 lap on
    # (verified from race data in racing_rl/analysis/)
    COMPLETABLE_MAPS = [
        "Catalunya", "Sepang", "Nuerburgring"  # confirmed lap_counts >= 1
    ]

    parser = argparse.ArgumentParser(description="D²PPO Stage 1: BC Pre-training from MPC")
    parser.add_argument("--maps", nargs="+", default=ALL_MAPS,
                        help="Maps to collect demos on (default: all)")
    parser.add_argument("--completable_only", action="store_true",
                        help="Only use maps the racing_rl agent can complete "
                             "(Catalunya, Sepang). Overrides --maps.")
    parser.add_argument("--num_agents", type=int, default=3,
                        help="Agents per env during collection")
    parser.add_argument("--steps_per_map", type=int, default=7000,
                        help="Env steps per map during collection")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Parallel processes for demo collection (default: min(num_maps, cpu_count))")
    parser.add_argument("--load", type=str, default=None,
                        help="Path to a saved demos .pt file (skip collection)")
    parser.add_argument("--save_demos", type=str, default="demos/expert_demos.pt",
                        help="Where to save collected demos for reuse")
    parser.add_argument("--pretrained_expert", type=str, default="../racing_rl/actor_val_best.pt",
                        help="Path to pretrained model (.pt) from racing_rl to use as expert "
                             "(default: None, uses MPC as expert)")
    parser.add_argument("--model_type", type=str, default="mamba2", choices=["lstm", "mamba2"],
                        help="Network architecture backbone (lstm or mamba2)")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--tbtt_length", type=int, default=512,
                        help="TBTT window length (number of timesteps per chunk)")
    parser.add_argument("--traj_batch_size", type=int, default=9,
                        help="Number of trajectories to process in parallel")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dispersive_lambda", type=float, default=0.5)
    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="models/actor/pretrained")
    parser.add_argument("--save_name", type=str, default="actor_pretrained.pt")
    parser.add_argument("--critic_epochs", type=int, default=30)
    parser.add_argument("--critic_lr", type=float, default=1e-4)
    parser.add_argument("--critic_save_dir", type=str, default="models/critic/pretrained")
    parser.add_argument("--critic_save_name", type=str, default="critic_pretrained.pt")
    parser.add_argument("--skip_critic", action="store_true",
                        help="Skip critic pretraining (actor only)")

    args = parser.parse_args()

    if args.completable_only:
        args.maps = COMPLETABLE_MAPS
        print(f"[Main] --completable_only: restricting to {args.maps}")

    if args.load and os.path.exists(args.load):
        print(f"[Main] Loading demos from {args.load}")
        demos = torch.load(args.load, weights_only=False)
        print(f"[Main] Loaded {len(demos)} demos")
    elif args.pretrained_expert:
        print(f"[Main] Using pretrained expert: {args.pretrained_expert}")
        demos = collect_demos_pretrained(
            maps=args.maps,
            pretrained_model_path=args.pretrained_expert,
            num_agents=args.num_agents,
            steps_per_map=args.steps_per_map,
        )
        os.makedirs(os.path.dirname(args.save_demos) or ".", exist_ok=True)
        torch.save(demos, args.save_demos)
        print(f"[Main] Demos saved to {args.save_demos}")
    else:
        demos = collect_demos(
            maps=args.maps,
            num_agents=args.num_agents,
            steps_per_map=args.steps_per_map,
            max_workers=args.max_workers,
        )
        os.makedirs(os.path.dirname(args.save_demos) or ".", exist_ok=True)
        torch.save(demos, args.save_demos)
        print(f"[Main] Demos saved to {args.save_demos}")

    model = pretrain(
        demos=demos,
        model_type=args.model_type,
        epochs=args.epochs,
        tbtt_length=args.tbtt_length,
        traj_batch_size=args.traj_batch_size,
        lr=args.lr,
        dispersive_lambda=args.dispersive_lambda,
        num_diffusion_steps=args.num_diffusion_steps,
        gradient_accumulation_steps=args.gradient_accumulation,
        save_dir=args.save_dir,
        save_name=args.save_name,
    )

    print("\n[Validation] Sampling actions from pretrained model (with temporal context) …")
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        from collections import defaultdict
        val_traj_dict = defaultdict(list)
        for d in demos:
            val_traj_dict[(d["map"], d.get("agent", 0))].append(d)
        first_key = next(iter(val_traj_dict))
        val_demos = sorted(val_traj_dict[first_key], key=lambda x: x.get("step", 0))
        val_scans = torch.stack([torch.from_numpy(d["scan"]) for d in val_demos]).unsqueeze(1).to(device)
        val_states = torch.stack([torch.from_numpy(d["state"]) for d in val_demos]).to(device)
        val_actions = torch.stack([torch.from_numpy(d["action"]) for d in val_demos]).to(device)
        T_val = val_scans.shape[0]
        warmup = min(T_val - 8, 200)  # build temporal context over 200 steps

        if args.model_type == "lstm":
            obs_buffer = model.create_observation_buffer(1, device)
            hidden_h, hidden_c = model.get_init_hidden(1, device, transpose=True)
        else:
            conv_state, ssm_state = model.allocate_state(1, device)

        for t in range(warmup):
            if args.model_type == "lstm":
                obs_feat, obs_buffer, hidden_h, hidden_c = model.encode_observation(
                    val_scans[t:t+1], val_states[t:t+1], obs_buffer, hidden_h, hidden_c)
            else:
                obs_feat, conv_state, ssm_state = model.encode_observation(
                    val_scans[t:t+1], val_states[t:t+1], conv_state, ssm_state)

        eval_start = warmup
        eval_end = min(eval_start + 8, T_val)
        sampled_list = []
        expert_list = []
        for t in range(eval_start, eval_end):
            if args.model_type == "lstm":
                obs_feat, obs_buffer, hidden_h, hidden_c = model.encode_observation(
                    val_scans[t:t+1], val_states[t:t+1], obs_buffer, hidden_h, hidden_c)
            else:
                obs_feat, conv_state, ssm_state = model.encode_observation(
                    val_scans[t:t+1], val_states[t:t+1], conv_state, ssm_state)
            act = model.sample_action(obs_feat, deterministic=True)
            sampled_list.append(act)
            expert_list.append(val_actions[t:t+1])

        sampled = torch.cat(sampled_list, dim=0)
        expert = torch.cat(expert_list, dim=0)

        print(f"  Expert actions (first 4): {expert[:4].cpu().numpy()}")
        print(f"  Model actions  (first 4): {sampled[:4].cpu().numpy()}")
        mse = F.mse_loss(sampled, expert).item()
        steer_mse = F.mse_loss(sampled[:, 0], expert[:, 0]).item()
        speed_mse = F.mse_loss(sampled[:, 1], expert[:, 1]).item()
        print(f"  Action MSE: {mse:.4f}  (steering: {steer_mse:.4f}, speed: {speed_mse:.4f})")

    actor_weights_path = os.path.join(args.save_dir, args.save_name)
    if not args.skip_critic:
        actor_sd = torch.load(actor_weights_path, weights_only=False)
        pretrain_critic(
            demos=demos,
            actor_state_dict=actor_sd,
            model_type=args.model_type,
            epochs=args.critic_epochs,
            tbtt_length=args.tbtt_length,
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
