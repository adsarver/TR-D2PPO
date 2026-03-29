"""
algorithm_grid_search.py – Exhaustive baseline parameter search for F1TENTH.

Uses pre-determined per-map best speeds (from map_speed_search.py) so that
each (algorithm, map) pair runs at its known-safe maximum speed.  The search
focuses purely on finding the parameter combination that minimises average
lap time across all maps.

Per-map evaluations are distributed across worker processes.
Results are written to CSV files under results/.
"""

import os
import sys
import csv
import time
import itertools
import multiprocessing as mp
from datetime import datetime

import gym
import numpy as np

from baselines.pure_pursuit import PurePursuit
from baselines.gap_follow import GapFollow
from baselines.gap_follow_pure_pursuit import (
    GapFollowPurePursuit, GFPP_MAP_SPEED_LOOKUP,
)
from baselines.mpc_agent import MPCAgent, MPC_MAP_SPEED_LOOKUP
from utils.utils import get_map_dir, generate_start_poses

# ──────────────────────────────────────────────────────────────────────
# Car physics (must match the simulator)
# ──────────────────────────────────────────────────────────────────────
PARAMS_DICT = {
    'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562,
    'lf': 0.15875, 'lr': 0.17145, 'h': 0.074,
    'm': 3.74, 'I': 0.04712,
    's_min': -0.34, 's_max': 0.34,
    'sv_min': -3.2, 'sv_max': 3.2,
    'v_switch': 7.319, 'a_max': 9.51,
    'v_min': -5.0, 'v_max': 20.0,
    'width': 0.31, 'length': 0.58,
}

WHEELBASE = PARAMS_DICT['lf'] + PARAMS_DICT['lr']
MAX_STEERING = PARAMS_DICT['s_max']
LIDAR_BEAMS = 1080
LIDAR_FOV = 4.7
NUM_AGENTS = 1  # single-agent evaluation

# Minimum laps an agent must complete on every map
MIN_LAPS = 3
# Per-run simulation timeout: generous enough for MIN_LAPS laps.
LAP_TIMEOUT = 70.0 * MIN_LAPS
# Collision duration before aborting (seconds).
COLLISION_TIMEOUT = 0.5
# Early-stop a param combo if it fails this many maps at the base speed.
EARLY_STOP_MAP_FAILURES = 3

# Multiprocessing: 0 = auto-detect.
NUM_WORKERS = 12  # i9-14900K: 32 threads available

# ──────────────────────────────────────────────────────────────────────
# Map list – every directory under maps/ that has a raceline CSV
# ──────────────────────────────────────────────────────────────────────
ALL_MAPS = sorted([
    d for d in os.listdir("maps")
    if os.path.isdir(os.path.join("maps", d))
    and os.path.isfile(os.path.join("maps", d, f"{d}_raceline.csv"))
    and d != "BrandsHatchObs"  # obstacle variant, skip for baseline search
])

# ──────────────────────────────────────────────────────────────────────
# Per-map speed lookups (from map_speed_search.py)
# ──────────────────────────────────────────────────────────────────────
DEFAULT_SPEED = 5.0  # fallback for maps / algos without a lookup entry

ALGO_SPEED_LOOKUPS = {
    "GapFollowPurePursuit": GFPP_MAP_SPEED_LOOKUP,
    "MPCAgent":             MPC_MAP_SPEED_LOOKUP,
}


def get_map_speed(algo_name: str, map_name: str) -> float:
    """Return the known-safe max speed for (algo, map), or DEFAULT_SPEED."""
    return ALGO_SPEED_LOOKUPS.get(algo_name, {}).get(map_name, DEFAULT_SPEED)

# ──────────────────────────────────────────────────────────────────────
# Parameter grids (max_speed is NOT here – it is swept separately)
# ──────────────────────────────────────────────────────────────────────
# Paper starting values → PurePursuit(lookahead=1.5, min_speed=1.0)
PP_GRID = {
    'lookahead_distance': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
    'min_speed':          [0.5, 1.0, 1.5],
}

# Paper starting values → GapFollow(bubble_radius=8, max_gap_safe_dist=1.8,
#   max_sight=8.0, min_speed=1.0, hysteresis=0.2)
GF_GRID = {
    'bubble_radius':     [5, 8, 12],
    'max_gap_safe_dist': [1.2, 1.8, 2.5],
    'max_sight':         [5.0, 8.0, 10.0],
}

# Paper starting values → GFPP(lookahead=1.5, threshold_v_min=0.9,
#   threshold_v_max=1.8)
GFPP_GRID = {
    'lookahead_distance':  [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
    'threshold_at_v_min':  [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
    'threshold_at_v_max':  [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.3, 2.5],
}

# Paper starting values → MPC(horizon=8, speed_scale=1.2, emergency_dist=1.2)
MPC_GRID = {
    'horizon':        [6, 7, 8, 9, 10, 11, 12],
    'speed_scale':    [0.8, 0.9, 1.0, 1.1, 1.2],
    'emergency_dist': [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
}


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def grid_combos(grid: dict) -> list:
    """Return every combination of values from *grid* as a list of dicts."""
    if not grid:
        return [{}]
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]


def make_agent(algo_name: str, map_name: str, max_speed: float, extra: dict):
    """Instantiate a fresh baseline agent for one map + param set."""
    if algo_name == "PurePursuit":
        return PurePursuit(
            map_name=map_name,
            wheelbase=WHEELBASE,
            max_steering=MAX_STEERING,
            max_speed=max_speed,
            **extra,
        )
    if algo_name == "GapFollow":
        return GapFollow(
            map_name=map_name,
            num_beams=LIDAR_BEAMS,
            fov=LIDAR_FOV,
            max_speed=max_speed,
            max_steering=MAX_STEERING,
            **extra,
        )
    if algo_name == "GapFollowPurePursuit":
        return GapFollowPurePursuit(
            map_name=map_name,
            wheelbase=WHEELBASE,
            max_steering=MAX_STEERING,
            max_speed=max_speed,
            num_beams=LIDAR_BEAMS,
            fov=LIDAR_FOV,
            **extra,
        )
    if algo_name == "MPCAgent":
        return MPCAgent(
            map_name=map_name,
            wheelbase=WHEELBASE,
            max_steering=MAX_STEERING,
            max_speed=max_speed,
            max_accel=PARAMS_DICT['a_max'],
            num_beams=LIDAR_BEAMS,
            fov=LIDAR_FOV,
            **extra,
        )
    raise ValueError(f"Unknown algorithm: {algo_name}")


# ──────────────────────────────────────────────────────────────────────
# Core: run MIN_LAPS laps on one map
# ──────────────────────────────────────────────────────────────────────

def run_laps(env, agent, map_name, max_speed, num_laps=MIN_LAPS):
    """
    Run *num_laps* consecutive laps with *agent* on *map_name*.

    Returns
    -------
    completed : bool
        True if the agent completed all *num_laps* laps.
    avg_lap_time : float
        Mean simulated seconds per lap (meaningful only when completed).
    """
    # Switch the environment to the target map
    env.update_map(get_map_dir(map_name) + f"/{map_name}_map", ".png")

    # Find a collision-free starting pose (retry with small offsets)
    obs = None
    for attempt in range(20):
        poses = generate_start_poses(map_name, NUM_AGENTS,
                                     race=True, race_offset=attempt * 0.15)
        obs, _, _, _ = env.reset(poses=poses)
        if obs['collisions'][0] == 0:
            break
    else:
        return False, 0.0  # no safe start found

    sim_time = 0.0
    collision_dur = 0.0
    current_lap = 0
    lap_start_time = 0.0
    lap_times = []

    while sim_time < LAP_TIMEOUT and current_lap < num_laps:
        # Stuck-in-collision abort
        if obs['collisions'][0] == 1:
            collision_dur += env.timestep
            if collision_dur > COLLISION_TIMEOUT:
                return False, 0.0
        else:
            collision_dur = 0.0

        # Get action from the agent
        action = agent.get_actions_batch(obs)[:NUM_AGENTS]  # (1, 2)
        # Enforce the speed cap on the action
        action[:, 1] = np.clip(action[:, 1], 0.0, max_speed)

        # Step the simulator
        next_obs, step_time, _, _ = env.step(action)
        sim_time += step_time

        # Lap completion check
        if next_obs['lap_counts'][0] > current_lap:
            lap_times.append(sim_time - lap_start_time)
            current_lap = next_obs['lap_counts'][0]
            lap_start_time = sim_time

        obs = next_obs

    if current_lap >= num_laps:
        return True, float(np.mean(lap_times))
    return False, 0.0  # timed-out or crashed before completing all laps


# ──────────────────────────────────────────────────────────────────────
# Worker: evaluate one (algo, params, speed, map) in a subprocess
# ──────────────────────────────────────────────────────────────────────

# Per-process cached environment (avoid re-creating for every map).
_worker_env = None


def _worker_init(work_dir):
    """Each pool worker creates its own gym environment eagerly."""
    global _worker_env
    # Spawned children may have a different cwd; restore it.
    os.chdir(work_dir)
    first_map = ALL_MAPS[0]
    _worker_env = gym.make(
        "f110_gym:f110-v0",
        map=get_map_dir(first_map) + f"/{first_map}_map",
        num_agents=NUM_AGENTS,
        num_beams=LIDAR_BEAMS,
        fov=LIDAR_FOV,
        params=PARAMS_DICT,
    )
    # reset + render so internal renderer state is populated before update_map
    poses = generate_start_poses(first_map, NUM_AGENTS, race=True)
    _worker_env.reset(poses=poses)
    _worker_env.render(mode="human")


def _evaluate_single_map(args):
    """
    Evaluate one (algo, params, speed, map) combo.
    Designed to be called via Pool.map / imap_unordered.

    Returns (combo_key, map_name, completed, avg_lap_time).
    """
    combo_key, algo_name, map_name, max_speed, extra_params = args
    env = _worker_env
    agent = make_agent(algo_name, map_name, max_speed, extra_params)
    ok, avg_lap = run_laps(env, agent, map_name, max_speed)
    return (combo_key, map_name, ok, avg_lap)


# ──────────────────────────────────────────────────────────────────────
# Main grid-search loop
# ──────────────────────────────────────────────────────────────────────

def _aggregate_combo_results(results_list, maps):
    """
    Given a list of (combo_key, map_name, ok, avg_lap) tuples for ONE combo,
    compute summary stats.
    """
    per_map = {}
    completed_times = []
    for _, map_name, ok, avg_lap in results_list:
        per_map[map_name] = (ok, avg_lap)
        if ok:
            completed_times.append(avg_lap)

    n_completed = sum(1 for ok, _ in per_map.values() if ok)
    completion_rate = n_completed / len(maps) if maps else 0.0
    avg_time = float(np.mean(completed_times)) if completed_times else float('inf')
    all_ok = n_completed == len(maps)
    return all_ok, completion_rate, avg_time, per_map


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    detail_csv  = os.path.join("results", f"grid_search_detail_{timestamp}.csv")
    summary_csv = os.path.join("results", f"grid_search_summary_{timestamp}.csv")

    # Determine worker count
    n_workers = NUM_WORKERS if NUM_WORKERS > 0 else min(os.cpu_count() / 2 or 1, 8)
    print(f"Launching pool with {n_workers} worker(s)  "
          f"({len(ALL_MAPS)} maps, {MIN_LAPS} laps per map)")

    work_dir = os.getcwd()
    pool = mp.Pool(processes=n_workers, initializer=_worker_init,
                   initargs=(work_dir,))

    algorithms = {
        "GapFollowPurePursuit": GFPP_GRID,
        "MPCAgent":             MPC_GRID,
    }

    # ── Build every (algo, params) combo ──
    all_combos = {}   # combo_key → (algo_name, extra_params)
    for algo_name, grid in algorithms.items():
        for ci, extra in enumerate(grid_combos(grid)):
            key = f"{algo_name}_{ci}"
            all_combos[key] = (algo_name, extra)

    total_combos = len(all_combos)
    total_items = total_combos * len(ALL_MAPS)
    print(f"\n  Total parameter combos across all baselines: {total_combos}")
    print(f"  Total work items: "
          f"{total_combos} × {len(ALL_MAPS)} maps = {total_items}")

    detail_fields = [
        "algorithm", "params",
        "completion_rate", "num_completed", "num_maps",
        "avg_lap_time", "per_map",
    ]
    df = open(detail_csv, "w", newline="")
    dw = csv.DictWriter(df, fieldnames=detail_fields)
    dw.writeheader()

    # ==================================================================
    #  Parameter search at per-map best speeds (all baselines at once)
    # ==================================================================
    print(f"\n{'=' * 70}")
    print(f"  Parameter search using per-map best speeds  "
          f"({MIN_LAPS} laps, {len(ALL_MAPS)} maps)")
    print(f"{'=' * 70}")

    # Build work items for ALL combos × ALL maps simultaneously
    work_items = []
    for key, (algo_name, extra) in all_combos.items():
        for map_name in ALL_MAPS:
            speed = get_map_speed(algo_name, map_name)
            work_items.append(
                (key, algo_name, map_name, speed, extra)
            )

    t0 = time.time()
    print(f"  Dispatching {len(work_items)} work items to {n_workers} workers...")

    # Use imap_unordered for streaming results + early-stop tracking
    combo_results = {key: [] for key in all_combos}   # key → list of results
    combo_failures = {key: 0 for key in all_combos}   # key → map fail count
    combo_stopped = set()  # combos that hit the early-stop threshold

    for result in pool.imap_unordered(_evaluate_single_map, work_items,
                                       chunksize=1):
        combo_key, map_name, ok, avg_lap = result
        combo_results[combo_key].append(result)
        if not ok:
            combo_failures[combo_key] += 1

        # Print progress every 50 results
        done = sum(len(v) for v in combo_results.values())
        if done % 50 == 0 or done == len(work_items):
            elapsed = time.time() - t0
            print(f"    [{done}/{len(work_items)}] "
                  f"{elapsed:.0f}s elapsed  "
                  f"({len(combo_stopped)} combos early-stopped)",
                  end="\r")

    print()  # newline after progress
    elapsed = time.time() - t0
    print(f"  Search complete in {elapsed:.1f}s")

    # ── Summarise Phase 1 results per combo ──
    combo_summaries = {}  # key → {algo, params, completion_rate, avg_lap}
    for key, (algo_name, extra) in all_combos.items():
        all_ok, comp_rate, avg_time, per_map = _aggregate_combo_results(
            combo_results[key], ALL_MAPS
        )
        combo_summaries[key] = {
            "algo": algo_name,
            "params": extra,
            "all_ok": all_ok,
            "completion_rate": comp_rate,
            "avg_lap_time": avg_time,
            "per_map": per_map,
        }
        # Log to CSV
        dw.writerow({
            "algorithm": algo_name,
            "params": str(extra),
            "completion_rate": f"{comp_rate:.3f}",
            "num_completed": sum(1 for ok, _ in per_map.values() if ok),
            "num_maps": len(ALL_MAPS),
            "avg_lap_time": f"{avg_time:.3f}",
            "per_map": str({
                m: (ok, f"{t:.2f}")
                for m, (ok, t) in per_map.items()
            }),
        })
    df.flush()

    # ── Pick best params per algorithm ──
    # Sort by: completion_rate DESC, then avg_lap_time ASC
    best_per_algo = {}
    for algo_name in algorithms:
        algo_combos = [
            (key, s) for key, s in combo_summaries.items()
            if s["algo"] == algo_name
        ]
        algo_combos.sort(
            key=lambda x: (-x[1]["completion_rate"], x[1]["avg_lap_time"])
        )
        if algo_combos:
            best_key, best_s = algo_combos[0]
            best_per_algo[algo_name] = {
                "params": best_s["params"],
                "avg_lap_time": best_s["avg_lap_time"],
                "completion_rate": best_s["completion_rate"],
            }
            pct = best_s['completion_rate'] * 100
            print(f"\n  {algo_name}: best params = {best_s['params']}")
            print(f"    completion={pct:.0f}%  avg_lap={best_s['avg_lap_time']:.2f}s")

            # Print top-3 for context
            for rank, (k, s) in enumerate(algo_combos[:3], 1):
                rpct = s['completion_rate'] * 100
                print(f"      #{rank} {s['params']}  "
                      f"comp={rpct:.0f}%  avg={s['avg_lap_time']:.2f}s")

    df.close()
    pool.close()
    pool.join()

    # ── Summary CSV ──
    with open(summary_csv, "w", newline="") as sf:
        sw = csv.DictWriter(sf, fieldnames=[
            "algorithm", "best_params",
            "best_avg_lap_time", "completion_rate",
        ])
        sw.writeheader()
        for algo, info in best_per_algo.items():
            sw.writerow({
                "algorithm":         algo,
                "best_params":       str(info["params"]),
                "best_avg_lap_time": f"{info['avg_lap_time']:.3f}",
                "completion_rate":   f"{info['completion_rate']:.3f}",
            })

    # ── Final console summary ──
    print(f"\n{'=' * 70}")
    print(f"  FINAL SUMMARY — Best Configuration per Baseline  "
          f"({MIN_LAPS}-lap requirement)")
    print(f"{'=' * 70}")
    for algo, info in best_per_algo.items():
        rate_pct = info['completion_rate'] * 100
        print(f"\n  {algo}:")
        print(f"    Avg lap time:    {info['avg_lap_time']:.2f} s")
        print(f"    Completion rate: {rate_pct:.0f}%")
        print(f"    Parameters:      {info['params']}")

    print(f"\n  Detail log : {detail_csv}")
    print(f"  Summary    : {summary_csv}")
    print()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
