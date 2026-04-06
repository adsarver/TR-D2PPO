"""
val.py — Validation: Race D2PPO vs GFPP & MPC on random generated tracks.
=========================================================================
Generates a fresh random track for each race, then pits the trained D2PPO
agent (ego, agent 0) against GapFollowPurePursuit (agent 1) and MPCAgent
(agent 2).  Tracks lap times, collisions, and prints a summary.

Usage:
    python val.py                                     # default settings
    python val.py --races 20 --laps 5
    python val.py --actor models/actor/best/actor_gen_500.pt
"""

import argparse
import os
import shutil
import time

import gym
import numpy as np
import torch

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

from D2PPO_agent import D2PPOAgent as PPOAgent
from baselines.gap_follow_pure_pursuit import GapFollowPurePursuit
from baselines.mpc_agent import MPCAgent
from track_generator import TrackGenerator
from utils.utils import get_map_dir, generate_start_poses

# ──────────────────────────────────────────────────────────────────────
# Car physics (must match train.py / simulator)
# ──────────────────────────────────────────────────────────────────────
PARAMS_DICT = {
    "mu": 1.0489, "C_Sf": 4.718, "C_Sr": 5.4562,
    "lf": 0.15875, "lr": 0.17145, "h": 0.074, "m": 3.74,
    "I": 0.04712, "s_min": -0.34, "s_max": 0.34,
    "sv_min": -3.2, "sv_max": 3.2, "v_switch": 7.319,
    "a_max": 9.51, "v_min": -5.0, "v_max": 20.0,
    "width": 0.31, "length": 0.58,
}
WHEELBASE = PARAMS_DICT["lf"] + PARAMS_DICT["lr"]
LIDAR_BEAMS = 1080
LIDAR_FOV = 4.7

# ──────────────────────────────────────────────────────────────────────
# Best baseline parameters (from grid search)
# ──────────────────────────────────────────────────────────────────────
BEST_GFPP_PARAMS = {
    "lookahead_distance": 1.4,
    "threshold_at_v_min": 1.0,
    "threshold_at_v_max": 2.5,
}
BEST_MPC_PARAMS = {
    "horizon": 8,
    "speed_scale": 0.8,
    "emergency_dist": 0.8,
}

# ──────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────
NUM_AGENTS = 3  # D2PPO, GFPP, MPC
AGENT_NAMES = ["D2PPO", "GFPP", "MPC"]
DEFAULT_ACTOR = "actor_gen_36.pt"
DEFAULT_CRITIC = 'models/actor/pretrained/actor_pretrained.pt'
DEFAULT_RACES = 10
DEFAULT_LAPS = 3
LAP_TIMEOUT = 300.0        # seconds per race before giving up
COLLISION_TIMEOUT = 0.5     # seconds stuck in collision → reset agent
BASELINE_SPEED = 12.0       # max speed for baselines on generated tracks


def parse_args():
    p = argparse.ArgumentParser(description="Validate D2PPO vs baselines on random tracks")
    p.add_argument("--actor", default=DEFAULT_ACTOR, help="Path to actor weights")
    p.add_argument("--critic", default=DEFAULT_CRITIC, help="Path to critic weights")
    p.add_argument("--races", type=int, default=DEFAULT_RACES, help="Number of races")
    p.add_argument("--laps", type=int, default=DEFAULT_LAPS, help="Laps per race")
    p.add_argument("--render", action="store_true", help="Render the environment")
    p.add_argument("--speed", type=float, default=BASELINE_SPEED,
                   help="Max speed for baselines on generated tracks")
    return p.parse_args()


def make_baselines(map_name, max_speed):
    """Create GFPP and MPC agents with best grid-search params."""
    gfpp = GapFollowPurePursuit(
        map_name=map_name,
        wheelbase=WHEELBASE,
        max_steering=PARAMS_DICT["s_max"],
        max_speed=max_speed,
        num_beams=LIDAR_BEAMS,
        fov=LIDAR_FOV,
        **BEST_GFPP_PARAMS,
    )
    mpc = MPCAgent(
        map_name=map_name,
        wheelbase=WHEELBASE,
        max_steering=PARAMS_DICT["s_max"],
        max_speed=max_speed,
        max_accel=PARAMS_DICT["a_max"],
        num_beams=LIDAR_BEAMS,
        fov=LIDAR_FOV,
        **BEST_MPC_PARAMS,
    )
    return gfpp, mpc


def run_race(env, d2ppo, gfpp, mpc, map_name, num_laps, render, max_speed):
    """
    Run a single multi-lap race.

    Returns
    -------
    results : list[dict]
        Per-agent dict with keys: completed, laps, lap_times, collisions.
    """
    poses = generate_start_poses(map_name, NUM_AGENTS, race=True)
    obs, _, _, _ = env.reset(poses=poses)

    if render:
        env.render(mode="human")

    lap_counts = np.zeros(NUM_AGENTS, dtype=int)
    lap_start = np.zeros(NUM_AGENTS)
    lap_times = [[] for _ in range(NUM_AGENTS)]
    collision_counts = np.zeros(NUM_AGENTS, dtype=int)
    collision_dur = np.zeros(NUM_AGENTS)
    sim_time = 0.0

    while sim_time < LAP_TIMEOUT:
        # --- D2PPO action (agent 0) ---
        scan_t, state_t = d2ppo._obs_to_tensors(obs)
        action_t, _, _ = d2ppo.get_action_and_value(
            scan_t, state_t, deterministic=True
        )
        d2ppo_action = action_t.cpu().numpy()  # (num_d2ppo_agents, 2)

        # --- Baseline actions (agents 1, 2) ---
        gfpp_action = gfpp.get_action(obs, agent_idx=1)
        mpc_action = mpc.get_action(obs, agent_idx=2)

        # Assemble action array for all agents
        actions = np.zeros((NUM_AGENTS, 2), dtype=np.float32)
        actions[0] = d2ppo_action[0]
        actions[1] = gfpp_action
        actions[2] = mpc_action

        # Speed cap
        actions[:, 1] = np.clip(actions[:, 1], 0.0, max_speed)

        next_obs, step_time, _, _ = env.step(actions)
        sim_time += step_time

        if render:
            env.render(mode="human")

        # --- Collision tracking ---
        for i in range(NUM_AGENTS):
            if next_obs["collisions"][i] == 1:
                collision_dur[i] += step_time
                if collision_dur[i] > COLLISION_TIMEOUT:
                    collision_counts[i] += 1
                    collision_dur[i] = 0.0
                    # Reset this agent
                    cur_poses = np.stack([
                        next_obs["poses_x"], next_obs["poses_y"],
                        next_obs["poses_theta"],
                    ], axis=1)
                    new_poses = generate_start_poses(
                        map_name, NUM_AGENTS, agent_poses=cur_poses)
                    next_obs, _, _, _ = env.reset(
                        poses=new_poses, agent_idxs=np.array([i]))
                    d2ppo.reset_buffers(np.array([i])) if i == 0 else None
            else:
                collision_dur[i] = 0.0

        # --- Lap tracking ---
        for i in range(NUM_AGENTS):
            new_laps = int(next_obs["lap_counts"][i])
            if new_laps > lap_counts[i]:
                lap_times[i].append(sim_time - lap_start[i])
                lap_start[i] = sim_time
                lap_counts[i] = new_laps

        # Check if all agents finished or all timed out
        all_done = all(lap_counts[i] >= num_laps for i in range(NUM_AGENTS))
        if all_done:
            break

        obs = next_obs

        # Print live status
        status_parts = []
        for i, name in enumerate(AGENT_NAMES):
            v = obs["linear_vels_x"][i]
            status_parts.append(f"{name}: {v:.1f} m/s lap {lap_counts[i]}/{num_laps}")
        print("  " + " | ".join(status_parts), end="\r")

    print()  # newline after carriage return

    results = []
    for i in range(NUM_AGENTS):
        completed = lap_counts[i] >= num_laps
        avg_lap = float(np.mean(lap_times[i])) if lap_times[i] else float("inf")
        results.append({
            "name": AGENT_NAMES[i],
            "completed": completed,
            "laps": int(lap_counts[i]),
            "lap_times": lap_times[i],
            "avg_lap_time": avg_lap,
            "collisions": int(collision_counts[i]),
        })
    return results


def main():
    args = parse_args()

    # --- Track Generator ---
    track_gen = TrackGenerator(
        min_track_length=150,
        max_track_length=1500,
        min_turns=6,
        max_turns=20,
        min_track_width=2.0,
        max_track_width=4.5,
        min_turn_radius=3.0,
        seed=None,
    )

    # --- Create initial env on a throwaway map (will be replaced) ---
    init_map = "Hockenheim"
    env = gym.make(
        "f110_gym:f110-v0",
        map=get_map_dir(init_map) + f"/{init_map}_map",
        num_agents=NUM_AGENTS,
        num_beams=LIDAR_BEAMS,
        fov=LIDAR_FOV,
        params=PARAMS_DICT,
    )
    poses = generate_start_poses(init_map, NUM_AGENTS, race=True)
    env.reset(poses=poses)
    if args.render:
        env.render(mode="human")

    # --- D2PPO Agent ---
    d2ppo = PPOAgent(
        num_agents=1,
        map_name=init_map,
        steps=256,
        params=PARAMS_DICT,
        transfer=[args.actor, args.critic],
    )

    # --- Race Loop ---
    print(f"\n{'=' * 70}")
    print(f"  VALIDATION — D2PPO vs GFPP vs MPC on random generated tracks")
    print(f"  Races: {args.races}   Laps per race: {args.laps}")
    print(f"  Actor: {args.actor}")
    print(f"  Baseline speed cap: {args.speed} m/s")
    print(f"{'=' * 70}\n")

    all_results = []  # list of per-race results
    last_track = None

    for race_idx in range(args.races):
        # Clean up previous generated track
        if last_track is not None:
            old_dir = os.path.join("maps", last_track)
            if os.path.isdir(old_dir):
                shutil.rmtree(old_dir, ignore_errors=True)

        # Generate a fresh track
        track_name = f"val_track_{race_idx}"
        track_gen.generate(track_name)
        last_track = track_name

        print(f"--- Race {race_idx + 1}/{args.races}  track={track_name} ---")

        # Switch environment to the new track
        env.sim.set_map(
            get_map_dir(track_name) + f"/{track_name}_map.yaml", ".png")
        if hasattr(env, 'renderer') and env.renderer is not None:
            env.renderer.update_map(
                get_map_dir(track_name) + f"/{track_name}_map", ".png")
        elif hasattr(type(env), 'renderer') and type(env).renderer is not None:
            type(env).renderer.update_map(
                get_map_dir(track_name) + f"/{track_name}_map", ".png")

        # Update D2PPO agent's waypoints
        wp_xy, wp_s, rl = d2ppo._load_waypoints(track_name)
        d2ppo.waypoints_xy = wp_xy
        d2ppo.waypoints_s = wp_s
        d2ppo.raceline_length = rl
        d2ppo.reset_buffers()

        # Create baselines for this track
        gfpp, mpc = make_baselines(track_name, args.speed)

        # Run the race
        race_results = run_race(
            env, d2ppo, gfpp, mpc, track_name, args.laps,
            args.render, args.speed)
        all_results.append(race_results)

        # Print race summary
        for r in race_results:
            status = "DONE" if r["completed"] else f"DNF ({r['laps']} laps)"
            avg = f"{r['avg_lap_time']:.2f}s" if r["completed"] else "N/A"
            print(f"  {r['name']:>6s}:  {status:<16s}  "
                  f"avg_lap={avg:<8s}  collisions={r['collisions']}")

    # Clean up last track
    if last_track is not None:
        old_dir = os.path.join("maps", last_track)
        if os.path.isdir(old_dir):
            shutil.rmtree(old_dir, ignore_errors=True)

    # ── Final summary ────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  FINAL SUMMARY  ({args.races} races, {args.laps} laps each)")
    print(f"{'=' * 70}")

    for agent_idx, name in enumerate(AGENT_NAMES):
        wins = 0
        finishes = 0
        total_avg_laps = []
        total_collisions = 0

        for race_results in all_results:
            r = race_results[agent_idx]
            total_collisions += r["collisions"]
            if r["completed"]:
                finishes += 1
                total_avg_laps.append(r["avg_lap_time"])
                # Check if this agent had the best time
                completed_times = [
                    rr["avg_lap_time"] for rr in race_results if rr["completed"]]
                if completed_times and r["avg_lap_time"] <= min(completed_times):
                    wins += 1

        overall_avg = (f"{np.mean(total_avg_laps):.2f}s"
                       if total_avg_laps else "N/A")
        print(f"\n  {name}:")
        print(f"    Wins:            {wins}/{args.races}")
        print(f"    Finishes:        {finishes}/{args.races}")
        print(f"    Avg lap time:    {overall_avg}")
        print(f"    Total collisions: {total_collisions}")

    print()
    env.close()


if __name__ == "__main__":
    main()
