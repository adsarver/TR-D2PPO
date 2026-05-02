import glob
import os
import pickle

import numpy as np


DEFAULT_AGENT_RENAMES = {
    "BC_LSTM": "BC-LSTM",
    "D2PPO": "D²PPO",
}


def load_race_data(path, agent_renames=None):
    with open(path, "rb") as file:
        race_data = pickle.load(file)
    normalize_agent_labels(race_data, agent_renames)
    return race_data


def normalize_agent_labels(race_data, agent_renames=None):
    for source, target in (agent_renames or DEFAULT_AGENT_RENAMES).items():
        if source in race_data and target not in race_data:
            race_data[target] = race_data.pop(source)
    return race_data


def load_centerline(map_name, maps_root="maps"):
    candidates = [
        os.path.join(maps_root, map_name, f"{map_name}_centerline.csv"),
        *glob.glob(os.path.join(maps_root, map_name, "*_centerline.csv")),
    ]

    centerline = None
    for path in candidates:
        try:
            centerline = np.genfromtxt(path, delimiter=",", comments="#")
            break
        except Exception:
            continue

    if centerline is None:
        raise FileNotFoundError(f"No centerline found for {map_name}")

    xy = centerline[:, :2]
    segment_lengths = np.sqrt((np.diff(xy, axis=0) ** 2).sum(axis=1))
    cumulative_s = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    lap_length = cumulative_s[-1] + np.linalg.norm(xy[-1] - xy[0])
    return xy, cumulative_s, lap_length


def project_progress(position, centerline_xy, centerline_s, lap_length):
    distances = np.sqrt(((centerline_xy - position) ** 2).sum(axis=1))
    closest_idx = int(np.argmin(distances))
    return float(centerline_s[closest_idx] / lap_length)


def new_lap_record():
    return {
        "Positions": [],
        "Velocity": [],
        "Time": None,
        "Collisions": 0,
        "WallCollisions": 0,
        "AgentCollisions": 0,
        "WallColSteps": 0,
        "AgentColSteps": 0,
    }


def format_lap_time(value):
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        if value == "DNF":
            return "DNF"
        return "—" if value is None else str(value)


def safe_name(name):
    return str(name).replace(" ", "_")


def map_results_dir(map_name=None):
    path = "analysis/map_results"
    if map_name is not None:
        path = os.path.join(path, map_name)
    os.makedirs(path, exist_ok=True)
    return path


def style_table(
    table,
    n_cols,
    n_rows,
    *,
    font_size=8,
    header_color="#2c3e50",
    stripe_color="#ecf0f1",
    scale=None,
):
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.auto_set_column_width(list(range(n_cols)))
    if scale is not None:
        table.scale(*scale)

    for col in range(n_cols):
        table[0, col].set_facecolor(header_color)
        table[0, col].set_text_props(color="white", fontweight="bold")

    for row in range(1, n_rows + 1):
        color = stripe_color if row % 2 == 0 else "white"
        for col in range(n_cols):
            table[row, col].set_facecolor(color)