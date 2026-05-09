"""Generate a CSV agent-selection plan for weight_initializer.py.

The plan is derived from paper race data, but it does not train from the
pickled observations directly.  Instead, it chooses which agent should be
re-run by weight_initializer on each map so fresh scan/state/action demos can
be collected.
"""

import argparse
import csv
import math
import os
import pickle
from pathlib import Path

import numpy as np


SIM_DT = 0.01


def _scalar(obs, key, agent_idx=0, default=0.0):
    value = obs.get(key, default)
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr)
    if agent_idx >= arr.shape[0]:
        return float(default)
    return float(arr[agent_idx])


def _agent_idx(observations):
    if not observations:
        return 0
    return int(observations[0].get("ego_idx", 0))


def _lap_times_from_trace(observations, agent_idx=0):
    lap_times = []
    prev_lap_time = 0.0
    for obs in observations:
        lap_time = obs.get("lap_time")
        if lap_time is None:
            continue
        lap_time = float(lap_time)
        if prev_lap_time > 0.5 and lap_time < prev_lap_time - 0.5:
            lap_times.append(prev_lap_time)
        prev_lap_time = lap_time

    col_exit = any(bool(obs.get("col_exit", False)) for obs in observations)
    if prev_lap_time > 0.5 and not col_exit:
        lap_times.append(prev_lap_time)
    saved_laps = max((_scalar(obs, "lap_counts", agent_idx, 0.0) for obs in observations), default=0.0)
    if not lap_times and saved_laps > 0:
        saved_lap_time = max((_scalar(obs, "lap_times", agent_idx, 0.0) for obs in observations), default=0.0)
        if saved_lap_time > 0.5:
            lap_times.append(saved_lap_time)
    return lap_times


def _score_trace(observations, expected_laps):
    agent_idx = _agent_idx(observations)
    lap_times = _lap_times_from_trace(observations, agent_idx)
    lap_count_completed = float(len(lap_times))
    saved_completed_laps = max((_scalar(obs, "lap_counts", agent_idx, 0.0) for obs in observations), default=0.0)
    completed_laps = max(0.0, lap_count_completed, saved_completed_laps)
    success_rate = min(completed_laps / max(float(expected_laps), 1e-6), 1.0)
    survival_steps = max(0, len(observations) - 1)
    survival_time = survival_steps * SIM_DT
    col_exit = any(bool(obs.get("col_exit", False)) for obs in observations)
    fastest_lap = min(lap_times) if lap_times else math.inf
    mean_lap = float(np.mean(lap_times)) if lap_times else math.inf
    wall_collision_steps = sum(int(_scalar(obs, "collisions", agent_idx, 0.0) == 1.0) for obs in observations)
    agent_collision_steps = sum(int(_scalar(obs, "collisions", agent_idx, 0.0) == 2.0) for obs in observations)
    return {
        "completed_laps": float(completed_laps),
        "success_rate": float(success_rate),
        "survival_steps": int(survival_steps),
        "survival_time": float(survival_time),
        "fastest_lap_time": float(fastest_lap),
        "mean_lap_time": float(mean_lap),
        "col_exit": bool(col_exit),
        "wall_collision_steps": int(wall_collision_steps),
        "agent_collision_steps": int(agent_collision_steps),
    }


def _fmt(value):
    if isinstance(value, float) and math.isinf(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return value


def _load_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def _select_no_opp_rows(path, expected_laps):
    race_data = _load_pickle(path)
    rows = []
    maps = sorted({map_name for agent_maps in race_data.values() for map_name in agent_maps})
    for map_name in maps:
        candidates = []
        for agent, agent_maps in race_data.items():
            observations = agent_maps.get(map_name)
            if not observations:
                continue
            stats = _score_trace(observations, expected_laps)
            if stats["completed_laps"] < expected_laps:
                continue
            if not math.isfinite(stats["fastest_lap_time"]):
                continue
            candidates.append((agent, stats))
        if not candidates:
            continue
        candidates.sort(key=lambda item: (
            item[1]["fastest_lap_time"],
            item[0],
        ))
        agent, stats = candidates[0]
        rows.append(_row("no_opp", map_name, agent, path, "target_laps_fastest_lap", stats))
    return rows


def _select_overtake_rows(path, expected_laps):
    race_data = _load_pickle(path)
    rows = []
    maps = sorted({map_name for agent_maps in race_data.values() for map_name in agent_maps})
    for map_name in maps:
        candidates = []
        for agent, agent_maps in race_data.items():
            observations = agent_maps.get(map_name)
            if not observations:
                continue
            stats = _score_trace(observations, expected_laps)
            if stats["col_exit"] and stats["completed_laps"] < expected_laps:
                continue
            candidates.append((agent, stats))
        if not candidates:
            continue
        candidates.sort(key=lambda item: (
            -item[1]["survival_time"],
            item[1]["fastest_lap_time"],
            -item[1]["success_rate"],
            item[0],
        ))
        agent, stats = candidates[0]
        rows.append(_row("overtake", map_name, agent, path, "survival_time_fastest_lap", stats))
    return rows


def _row(scenario, map_name, agent, source_path, selection_rule, stats):
    row = {
        "scenario": scenario,
        "map": map_name,
        "agent": agent,
        "selection_rule": selection_rule,
        "source_pickle": source_path,
    }
    row.update(stats)
    return row


def generate_plan(no_opp_path, overtake_path, output_path, expected_laps, overtake_expected_laps):
    rows = []
    rows.extend(_select_no_opp_rows(no_opp_path, expected_laps))
    rows.extend(_select_overtake_rows(overtake_path, overtake_expected_laps))

    fieldnames = [
        "scenario",
        "map",
        "agent",
        "selection_rule",
        "success_rate",
        "completed_laps",
        "survival_steps",
        "survival_time",
        "fastest_lap_time",
        "mean_lap_time",
        "col_exit",
        "wall_collision_steps",
        "agent_collision_steps",
        "source_pickle",
    ]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(row.get(key, "")) for key in fieldnames})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate a weight_initializer agent-selection CSV from paper data")
    parser.add_argument("--no_opp", default="analysis/analysis/race_data_CS677_no_opp.pkl")
    parser.add_argument("--overtake", default="analysis/analysis/race_data_CS677_opp.pkl")
    parser.add_argument("--output", default="analysis/analysis/weight_initializer_agent_plan.csv")
    parser.add_argument("--expected_laps", type=float, default=3.0)
    parser.add_argument("--overtake_expected_laps", type=float, default=None)
    args = parser.parse_args()

    overtake_expected_laps = args.expected_laps if args.overtake_expected_laps is None else args.overtake_expected_laps
    rows = generate_plan(
        no_opp_path=args.no_opp,
        overtake_path=args.overtake,
        output_path=args.output,
        expected_laps=args.expected_laps,
        overtake_expected_laps=overtake_expected_laps,
    )
    counts = {}
    for row in rows:
        counts[row["scenario"]] = counts.get(row["scenario"], 0) + 1
    print(f"Wrote {len(rows)} rows to {args.output}")
    for scenario, count in sorted(counts.items()):
        print(f"  {scenario}: {count}")


if __name__ == "__main__":
    main()