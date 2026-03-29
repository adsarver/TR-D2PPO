"""
map_speed_search.py – Per-map maximum-speed search for F1TENTH baselines.

Uses the best parameters from a prior grid search (summary CSV) and, for
each algorithm × map, escalates max_speed until the agent can no longer
complete 3 laps.  Optionally also sweeps min_speed downward.

Results are written to results/map_speed_results_<timestamp>.csv.
"""

import os
import sys
import csv
import time
import multiprocessing as mp
from datetime import datetime

import gym
import numpy as np

from baselines.gap_follow_pure_pursuit import GapFollowPurePursuit
from baselines.mpc_agent import MPCAgent
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
NUM_AGENTS = 1

MIN_LAPS = 3
LAP_TIMEOUT = 70.0 * MIN_LAPS
COLLISION_TIMEOUT = 0.5

# Speed search range
SPEED_MIN = 3.0
SPEED_MAX = 20.0
SPEED_STEP = 0.5
# After this many consecutive failed speeds, stop escalating for a map.
MAX_CONSECUTIVE_FAILS = 3

NUM_WORKERS = 12

# ──────────────────────────────────────────────────────────────────────
# Best parameters from the grid search
# ──────────────────────────────────────────────────────────────────────
BEST_PARAMS = {
    "GapFollowPurePursuit": {
        'lookahead_distance': 1.4,
        'threshold_at_v_min': 1.0,
        'threshold_at_v_max': 2.5,
    },
    "MPCAgent": {
        'horizon': 8,
        'speed_scale': 0.8,
        'emergency_dist': 0.8,
    },
}

# ──────────────────────────────────────────────────────────────────────
# Map list
# ──────────────────────────────────────────────────────────────────────
ALL_MAPS = sorted([
    d for d in os.listdir("maps")
    if os.path.isdir(os.path.join("maps", d))
    and os.path.isfile(os.path.join("maps", d, f"{d}_raceline.csv"))
    and d != "BrandsHatchObs"
])


# ──────────────────────────────────────────────────────────────────────
# Agent construction
# ──────────────────────────────────────────────────────────────────────
def make_agent(algo_name, map_name, max_speed, min_speed, extra):
    if algo_name == "GapFollowPurePursuit":
        return GapFollowPurePursuit(
            map_name=map_name,
            wheelbase=WHEELBASE,
            max_steering=MAX_STEERING,
            max_speed=max_speed,
            min_speed=min_speed,
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
            min_speed=min_speed,
            max_accel=PARAMS_DICT['a_max'],
            num_beams=LIDAR_BEAMS,
            fov=LIDAR_FOV,
            **extra,
        )
    raise ValueError(f"Unknown algorithm: {algo_name}")


# ──────────────────────────────────────────────────────────────────────
# Core: run MIN_LAPS on one map
# ──────────────────────────────────────────────────────────────────────
def run_laps(env, agent, map_name, max_speed, num_laps=MIN_LAPS):
    env.update_map(get_map_dir(map_name) + f"/{map_name}_map", ".png")

    obs = None
    for attempt in range(20):
        poses = generate_start_poses(map_name, NUM_AGENTS,
                                     race=True, race_offset=attempt * 0.15)
        obs, _, _, _ = env.reset(poses=poses)
        if obs['collisions'][0] == 0:
            break
    else:
        return False, 0.0

    sim_time = 0.0
    collision_dur = 0.0
    current_lap = 0
    lap_start_time = 0.0
    lap_times = []

    while sim_time < LAP_TIMEOUT and current_lap < num_laps:
        if obs['collisions'][0] == 1:
            collision_dur += env.timestep
            if collision_dur > COLLISION_TIMEOUT:
                return False, 0.0
        else:
            collision_dur = 0.0

        action = agent.get_actions_batch(obs)[:NUM_AGENTS]
        action[:, 1] = np.clip(action[:, 1], 0.0, max_speed)
        next_obs, step_time, _, _ = env.step(action)
        sim_time += step_time

        if next_obs['lap_counts'][0] > current_lap:
            lap_times.append(sim_time - lap_start_time)
            current_lap = next_obs['lap_counts'][0]
            lap_start_time = sim_time

        obs = next_obs

    if current_lap >= num_laps:
        return True, float(np.mean(lap_times))
    return False, 0.0


# ──────────────────────────────────────────────────────────────────────
# Worker pool
# ──────────────────────────────────────────────────────────────────────
_worker_env = None


def _worker_init(work_dir):
    global _worker_env
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
    poses = generate_start_poses(first_map, NUM_AGENTS, race=True)
    _worker_env.reset(poses=poses)
    _worker_env.render(mode="human")


def _evaluate(args):
    """Evaluate (algo, map, max_speed, min_speed).
    Returns (algo, map, max_speed, min_speed, ok, avg_lap_time).
    """
    algo_name, map_name, max_speed, min_speed, extra = args
    agent = make_agent(algo_name, map_name, max_speed, min_speed, extra)
    ok, avg_lap = run_laps(_worker_env, agent, map_name, max_speed)
    return (algo_name, map_name, max_speed, min_speed, ok, avg_lap)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join("results", f"map_speed_results_{timestamp}.csv")

    n_workers = NUM_WORKERS if NUM_WORKERS > 0 else max(1, (os.cpu_count() or 2) // 2)

    speeds = np.arange(SPEED_MIN, SPEED_MAX + SPEED_STEP / 2, SPEED_STEP)
    speeds = [round(float(s), 1) for s in speeds]

    # Build ALL work items: algo × map × speed
    work_items = []
    for algo_name, extra in BEST_PARAMS.items():
        for map_name in ALL_MAPS:
            for spd in speeds:
                work_items.append((algo_name, map_name, spd, 1.0, extra))

    total = len(work_items)
    print(f"Per-map speed search: {len(BEST_PARAMS)} algos × "
          f"{len(ALL_MAPS)} maps × {len(speeds)} speeds = {total} evaluations")
    print(f"Workers: {n_workers}  |  Laps per eval: {MIN_LAPS}")
    print(f"Speed range: {SPEED_MIN}..{SPEED_MAX} step {SPEED_STEP}")
    print(f"Output: {out_csv}\n")

    work_dir = os.getcwd()
    pool = mp.Pool(processes=n_workers, initializer=_worker_init,
                   initargs=(work_dir,))

    # Collect results
    # results_by[algo][map] = [(speed, ok, avg_lap), ...]
    results_by = {algo: {m: [] for m in ALL_MAPS} for algo in BEST_PARAMS}

    t0 = time.time()
    done_count = 0
    for result in pool.imap_unordered(_evaluate, work_items, chunksize=1):
        algo, map_name, max_spd, min_spd, ok, avg_lap = result
        results_by[algo][map_name].append((max_spd, ok, avg_lap))
        done_count += 1
        if done_count % 20 == 0 or done_count == total:
            elapsed = time.time() - t0
            pct = done_count / total * 100
            print(f"  [{done_count}/{total}] {pct:.0f}%  "
                  f"elapsed {elapsed:.0f}s", end="\r")

    pool.close()
    pool.join()
    elapsed = time.time() - t0
    print(f"\n  All evaluations complete in {elapsed:.1f}s\n")

    # ── Analyse: per map, find the max speed where all laps completed ──
    fields = [
        "algorithm", "map", "best_max_speed", "avg_lap_time_at_best",
        "all_speeds_tested", "pass_fail_summary",
    ]
    rows = []

    for algo in BEST_PARAMS:
        print(f"{'=' * 60}")
        print(f"  {algo}")
        print(f"{'=' * 60}")
        for map_name in ALL_MAPS:
            trials = sorted(results_by[algo][map_name], key=lambda t: t[0])
            best_speed = 0.0
            best_lap = float('inf')
            pf_summary = {}
            for spd, ok, avg_lap in trials:
                pf_summary[spd] = "PASS" if ok else "FAIL"
                if ok and spd > best_speed:
                    best_speed = spd
                    best_lap = avg_lap

            status = f"{best_speed:5.1f} m/s  avg_lap={best_lap:7.2f}s" if best_speed > 0 else "  FAILED at all speeds"
            print(f"    {map_name:20s}  {status}")

            rows.append({
                "algorithm": algo,
                "map": map_name,
                "best_max_speed": f"{best_speed:.1f}" if best_speed > 0 else "FAIL",
                "avg_lap_time_at_best": f"{best_lap:.3f}" if best_speed > 0 else "",
                "all_speeds_tested": str(
                    {s: f"{'PASS' if ok else 'FAIL'} {t:.2f}s"
                     for s, ok, t in trials}
                ),
                "pass_fail_summary": str(pf_summary),
            })
        print()

    # Write CSV
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    # ── Compact summary table ──
    print(f"\n{'=' * 70}")
    print(f"  COMPACT SUMMARY — Best max_speed per algorithm × map")
    print(f"{'=' * 70}")
    header = f"  {'Map':20s}"
    for algo in BEST_PARAMS:
        short = algo[:8]
        header += f"  {short:>12s}"
    print(header)
    print("  " + "-" * (20 + 14 * len(BEST_PARAMS)))

    for map_name in ALL_MAPS:
        line = f"  {map_name:20s}"
        for algo in BEST_PARAMS:
            trials = sorted(results_by[algo][map_name], key=lambda t: t[0])
            best_speed = 0.0
            for spd, ok, avg_lap in trials:
                if ok and spd > best_speed:
                    best_speed = spd
            cell = f"{best_speed:.1f}" if best_speed > 0 else "FAIL"
            line += f"  {cell:>12s}"
        print(line)

    print(f"\n  Results saved to: {out_csv}")
    print()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
