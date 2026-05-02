"""Three-way comparison: BC-LSTM vs BC-Diffusion vs RL-Diffusion.

Races all three actors on the same set of maps and pickles the
per-step trajectories.  Uses *snapshotted* weight files in
``analysis/snapshots/`` so the live training run cannot overwrite
the checkpoints mid-evaluation.

Snapshots (created with ``cp`` before running this script):
    analysis/snapshots/bc_lstm_snapshot.pt   ← racing_rl actor_val_best.pt
    analysis/snapshots/d2ppo_bc_snapshot.pt  ← models/actor/pretrained/actor_pretrained.pt
    analysis/snapshots/d2ppo_rl_snapshot.pt  ← actor_best2.pt (or current best)

Output pickle keys:
    "BC_LSTM"       — supervised LSTM baseline (racing_rl)
    "D2PPO_BC"      — diffusion policy after Stage-1 BC pretraining only
    "D2PPO_RL"      — diffusion policy after Stage-2 RL fine-tuning
"""

import os
import sys

# Make the repo root importable regardless of where this is launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import time

import gym
import numpy as np

from D2PPO_agent import D2PPOAgent
from analysis.bc_lstm_agent import BCLSTMAgent
from baselines.pure_pursuit import PurePursuit
from utils.utils import generate_start_poses, get_map_dir


# --------------------------------------------------------------------- #
# Snapshot paths — copies frozen at evaluation time so the running     #
# training job cannot overwrite them.                                   #
# --------------------------------------------------------------------- #
SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")
SNAPSHOTS = {
    "BC_LSTM":   os.path.join(SNAPSHOT_DIR, "bc_lstm_snapshot.pt"),
    "D2PPO_BC":  os.path.join(SNAPSHOT_DIR, "d2ppo_bc_snapshot.pt"),
    "D2PPO_RL":  os.path.join(SNAPSHOT_DIR, "d2ppo_rl_snapshot.pt"),
}
for label, p in SNAPSHOTS.items():
    if not os.path.isfile(p):
        raise FileNotFoundError(
            f"Missing snapshot for {label}: {p}\n"
            "Run:\n"
            "  cp actor_best2.pt analysis/snapshots/d2ppo_rl_snapshot.pt\n"
            "  cp models/actor/pretrained/actor_pretrained.pt analysis/snapshots/d2ppo_bc_snapshot.pt\n"
            "  cp /home/WVU-AD/ads00024/racing_rl/actor_val_best.pt analysis/snapshots/bc_lstm_snapshot.pt"
        )


# --------------------------------------------------------------------- #
# Race configuration                                                    #
# --------------------------------------------------------------------- #
NUM_AGENTS_TEST = 1
NUM_OVERTAKE_AGENTS = 0          # Solo evaluation — pure performance metric
NUM_AGENTS = NUM_OVERTAKE_AGENTS + NUM_AGENTS_TEST

LAPS_PER_RACE = 3
LAP_TIMEOUT_S = 200.0
COLLISION_EXIT_S = 0.50          # bail if stuck in collision longer than this

LIDAR_BEAMS = 1080
LIDAR_FOV = 4.7

# Set to a list of map names to restrict; ``None`` = all maps in maps/
# Catalunya / Nuerburgring / IMS match the previous paper's eval set.
EVAL_MAPS = ["Catalunya", "Nuerburgring", "IMS", "BrandsHatch", "Monza", "Spielberg"]
# EVAL_MAPS = None

NEURAL_AGENT_TYPES = (D2PPOAgent, BCLSTMAgent)

PARAMS_DICT = {
    'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562,
    'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74,
    'I': 0.04712, 's_min': -0.34, 's_max': 0.34,
    'sv_min': -3.2, 'sv_max': 3.2, 'v_switch': 7.319,
    'a_max': 9.51, 'v_min': -5.0, 'v_max': 20.0,
    'width': 0.31, 'length': 0.58,
}


# --------------------------------------------------------------------- #
# Map list                                                              #
# --------------------------------------------------------------------- #
ALL_MAPS = sorted(os.listdir(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "maps")))
MAPS = EVAL_MAPS if EVAL_MAPS is not None else ALL_MAPS
MAPS = [m for m in MAPS if m in ALL_MAPS]
INIT_MAP = MAPS[0]

print(f"[3way] Racing {len(MAPS)} maps: {MAPS}")
print(f"[3way] Snapshots: {list(SNAPSHOTS.keys())}")


# --------------------------------------------------------------------- #
# Environment                                                           #
# --------------------------------------------------------------------- #
env = gym.make(
    "f110_gym:f110-v0",
    map=get_map_dir(INIT_MAP) + f"/{INIT_MAP}_map",
    num_agents=NUM_AGENTS,
    num_beams=LIDAR_BEAMS,
    fov=LIDAR_FOV,
    params=PARAMS_DICT,
    show_agent_ids=True,
)

INITIAL_POSES = generate_start_poses(INIT_MAP, NUM_AGENTS, race=True)
obs, _, _, _ = env.reset(poses=INITIAL_POSES)
env.render(mode='human')

# Background-agent driver (only used if NUM_OVERTAKE_AGENTS > 0)
slow_pp = PurePursuit(
    map_name=INIT_MAP,
    lookahead_distance=1.5,
    wheelbase=PARAMS_DICT['lf'] + PARAMS_DICT['lr'],
    max_steering=PARAMS_DICT['s_max'],
    max_speed=8.0,
    min_speed=2.0,
)


# --------------------------------------------------------------------- #
# Race loop                                                             #
# --------------------------------------------------------------------- #
RACE_DATA = {}
failed_maps = set()


def race(agent, map_name, label, lap_count=LAPS_PER_RACE, speed_cap=None):
    """Run *agent* on *map_name* and append per-step obs to RACE_DATA[label][map_name]."""
    in_collision = True
    env.update_map(get_map_dir(map_name) + f"/{map_name}_map", ".png")
    if not isinstance(agent, NEURAL_AGENT_TYPES):
        agent.update_map(map_name)

    offset = 0.1
    while in_collision:
        poses = generate_start_poses(
            map_name, NUM_AGENTS, race=True, race_offset=offset)
        offset += 0.1
        obs, _, _, _ = env.reset(poses=poses)
        in_collision = obs['collisions'][0] != 0
        env.render(mode='human')

    if isinstance(agent, NEURAL_AGENT_TYPES):
        agent.reset_buffers()

    print(f"\n--- {label} Starting Lap 0 / {lap_count} on {map_name} ---")
    obs['col_exit'] = False
    RACE_DATA[label][map_name] = [obs]

    sim_time = 0.0
    lap_start_sim_time = 0.0
    current_lap = 0
    collision_sim_time = 0.0

    while current_lap < lap_count and (sim_time - lap_start_sim_time) < LAP_TIMEOUT_S:
        if obs['collisions'][0] == 1:
            collision_sim_time += env.timestep
            if collision_sim_time > COLLISION_EXIT_S:
                obs['col_exit'] = True
                RACE_DATA[label][map_name].append(obs)
                print(f"\n--- {label} stuck in collision on {map_name}, exiting ---")
                if current_lap == 0:
                    failed_maps.add(map_name + '-' + label)
                break
        else:
            collision_sim_time = 0.0

        # Action
        if not isinstance(agent, NEURAL_AGENT_TYPES):
            action_np = [agent.get_actions_batch(obs)[0]]
        else:
            scan_t, state_t = agent._obs_to_tensors(obs)
            action_t, _, _ = agent.get_action_and_value(
                scan_t, state_t, deterministic=True)
            action_np = action_t.cpu().numpy()

        if NUM_OVERTAKE_AGENTS > 0:
            action_np = np.concatenate(
                [action_np, slow_pp.get_actions_batch(obs)[NUM_AGENTS_TEST:]],
                axis=0,
            )
        if speed_cap is not None:
            action_np[..., 1] = np.clip(action_np[..., 1], None, speed_cap)

        next_obs, step_time, _, _ = env.step(action_np)
        sim_time += step_time
        lap_time = sim_time - lap_start_sim_time
        next_obs['col_exit'] = False
        next_obs['lap_time'] = lap_time

        if next_obs['lap_counts'][0] > current_lap:
            current_lap = next_obs['lap_counts'][0]
            print(f"\n--- {label} Completed Lap {current_lap} on "
                  f"{map_name} in {lap_time:.2f}s ---")
            lap_start_sim_time = sim_time

        obs = next_obs.copy()
        del next_obs['scans']
        RACE_DATA[label][map_name].append(next_obs)

        print(f"{lap_time:.2f}s: Speed: {next_obs['linear_vels_x'][0]:.2f}",
              end='\r')


# --------------------------------------------------------------------- #
# Build agents                                                          #
# --------------------------------------------------------------------- #
def build_bc_lstm():
    return BCLSTMAgent(
        num_agents=NUM_AGENTS_TEST,
        weights_path=SNAPSHOTS["BC_LSTM"],
    )


def build_diffusion(weights_path):
    """Construct a D2PPOAgent loaded from a snapshot and put it in deploy mode."""
    agent = D2PPOAgent(
        num_agents=NUM_AGENTS_TEST,
        map_name=INIT_MAP,
        steps=None,
        params=PARAMS_DICT,
        transfer=[weights_path, None],
    )
    # 5 DDIM steps, no action repeat, no torch.compile during data
    # collection (keeps startup fast and avoids recompiles per map).
    agent.deploy(action_repeat=0, ddim_steps=5, compile_model=False)
    return agent


# Order: cheapest first so failures show up quickly
AGENTS_TO_RACE = [
    ("BC_LSTM",  build_bc_lstm()),
    ("D2PPO_BC", build_diffusion(SNAPSHOTS["D2PPO_BC"])),
    ("D2PPO_RL", build_diffusion(SNAPSHOTS["D2PPO_RL"])),
]


# --------------------------------------------------------------------- #
# Run                                                                   #
# --------------------------------------------------------------------- #
for label, agent in AGENTS_TO_RACE:
    RACE_DATA[label] = {}
    for map_name in MAPS:
        race(agent, map_name, label=label, lap_count=LAPS_PER_RACE)

# --------------------------------------------------------------------- #
# Save                                                                  #
# --------------------------------------------------------------------- #
out_dir = os.path.join(os.path.dirname(__file__), "analysis")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"three_way_compare_{int(time.time())}.pkl")
with open(out_path, "wb") as f:
    pickle.dump(RACE_DATA, f)

good_maps = [
    m for m in MAPS
    if all(m + '-' + lbl not in failed_maps for lbl, _ in AGENTS_TO_RACE)
]
print(f"\n[3way] Saved → {out_path}")
print(f"[3way] Failed (any agent): {sorted(failed_maps)}")
print(f"[3way] Clean on all 3: {good_maps}")


# --------------------------------------------------------------------- #
# Quick CR@K + lap-time summary                                         #
# --------------------------------------------------------------------- #
print("\n--- Summary (CR@K=3 / mean lap time) ---")
print(f"{'map':<16} {'BC_LSTM':>20} {'D2PPO_BC':>20} {'D2PPO_RL':>20}")
for m in MAPS:
    cells = []
    for label, _ in AGENTS_TO_RACE:
        traj = RACE_DATA[label].get(m, [])
        if not traj:
            cells.append("--")
            continue
        last = traj[-1]
        col_exit = bool(last.get('col_exit', False))
        lap_counts = int(last['lap_counts'][0]) if 'lap_counts' in last else 0
        completed = (lap_counts >= LAPS_PER_RACE) and not col_exit

        # Mean lap time across completed laps
        lap_times = []
        prev_lap = 0
        prev_t = 0.0
        for o in traj[1:]:
            lc = int(o.get('lap_counts', [0])[0])
            t = float(o.get('lap_time', 0.0))
            if lc > prev_lap:
                # Lap completed — t is time within this new lap (resets at lap boundary)
                lap_times.append(t if prev_t < t else prev_t)
                prev_lap = lc
            prev_t = t
        mean_lt = (sum(lap_times) / len(lap_times)) if lap_times else float('nan')

        if completed:
            cells.append(f"OK  {mean_lt:6.2f}s")
        elif col_exit:
            cells.append(f"DNF (col, lap {lap_counts})")
        else:
            cells.append(f"DNF (timeout, lap {lap_counts})")
    print(f"{m:<16} {cells[0]:>20} {cells[1]:>20} {cells[2]:>20}")
